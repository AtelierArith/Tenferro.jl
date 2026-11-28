module TenferroNNlibExt

using NNlib
using Tenferro: Tenferro, EagerScalar, _ADTensor, _FloatTensor, _select, _scalar_result, _scalar_tensor_value

const B = Base.Broadcast

# Activations ------------------------------------------------------------------------
#
# Each activation is composed from tenferro-rs kernels. Kinks use `select`, so
# derivatives at the kinks follow NNlib's ChainRules (e.g. relu'(0) == 0).

_c(x, v) = convert(eltype(x), v)
_gt(x, v) = B.broadcasted(>, x, _c(x, v))
_lt(x, v) = B.broadcasted(<, x, _c(x, v))
_ge(x, v) = B.broadcasted(>=, x, _c(x, v))
_sel(c, t, f) = _select(c, t, f)
_mul(x, y) = B.broadcasted(*, x, y)
_add(x, y) = B.broadcasted(+, x, y)
_sub(x, y) = B.broadcasted(-, x, y)
# `x` for x <= 0 and 0 otherwise: keeps `exp` of an unselected branch finite.
_nonpos(x) = _sel(B.broadcasted(<=, x, zero(eltype(x))), x, 0)
_tanh(x) = B.broadcasted(tanh, x)
_sigmoid(x) = Tenferro._sigmoid(x)
_softplus(x) = Tenferro._softplus(x)

_relu(x) = _sel(_gt(x, 0), x, 0)
_relu6(x) = _sel(_gt(x, 0), _sel(_lt(x, 6), x, _c(x, 6)), 0)
_leakyrelu(x, a=NNlib.leakyrelu_a) = _sel(_gt(x, 0), x, _mul(x, _c(x, a)))
_elu(x, α=1) = _sel(_ge(x, 0), x, _mul(B.broadcasted(expm1, _nonpos(x)), _c(x, α)))
_celu(x, α=1) = _sel(_ge(x, 0), x, _mul(B.broadcasted(expm1, B.broadcasted(/, _nonpos(x), _c(x, α))), _c(x, α)))
function _selu(x)
    λ, α = _c(x, NNlib.selu_λ), _c(x, NNlib.selu_α)
    _mul(_sel(_gt(x, 0), x, _mul(B.broadcasted(expm1, _nonpos(x)), α)), λ)
end
_hardsigmoid(x) = _sel(_gt(x, -3), _sel(_lt(x, 3), B.broadcasted(/, _add(x, _c(x, 3)), _c(x, 6)), 1), 0)
# NNlib's hardswish derivative is x/3 + 1/2 on the closed interval [-3, 3].
_hardswish(x) = _mul(x, _sel(_lt(x, -3), 0, _sel(_gt(x, 3), 1, B.broadcasted(/, _add(x, _c(x, 3)), _c(x, 6)))))
_hardtanh(x) = _sel(_gt(x, -1), _sel(_lt(x, 1), x, _c(x, 1)), _c(x, -1))
_logsigmoid(x) = B.broadcasted(-, _softplus(B.broadcasted(-, x)))
_softsign(x) = B.broadcasted(/, x, _add(B.broadcasted(abs, x), one(eltype(x))))
_mish(x) = _mul(x, _tanh(_softplus(x)))
_lisht(x) = _mul(x, _tanh(x))
_tanhshrink(x) = _sub(x, _tanh(x))
_logcosh(x) = _sub(_add(x, _softplus(_mul(x, _c(x, -2)))), _c(x, log(2)))
function _softshrink(x, λ=0.5)
    l = _c(x, λ)
    _sel(_gt(x, l), _sub(x, l), _sel(_lt(x, -l), _add(x, l), 0))
end
_trelu(x, θ=1) = _sel(_gt(x, θ), x, 0)
function _gelu_sigmoid(x)
    α, λλ = _c(x, 0.044715), _c(x, NNlib.gelu_2λ)
    inner = _mul(_mul(x, λλ), _add(_mul(_mul(x, x), α), one(eltype(x))))
    _mul(x, _sigmoid(inner))
end
_quick_gelu(x) = _mul(x, _sigmoid(_mul(x, _c(x, 1.702))))

const UNARY = (
    (NNlib.relu, _relu), (NNlib.relu6, _relu6), (NNlib.leakyrelu, _leakyrelu),
    (NNlib.elu, _elu), (NNlib.celu, _celu), (NNlib.selu, _selu),
    (NNlib.gelu_tanh, Tenferro._gelu_tanh), (NNlib.gelu_erf, Tenferro._gelu_erf),
    (NNlib.gelu_sigmoid, _gelu_sigmoid), (NNlib.quick_gelu, _quick_gelu),
    (NNlib.swish, Tenferro._silu), (NNlib.hardswish, _hardswish),
    (NNlib.sigmoid, _sigmoid), (NNlib.sigmoid_fast, _sigmoid), (NNlib.hardσ, _hardsigmoid),
    (NNlib.logσ, _logsigmoid), (NNlib.hardtanh, _hardtanh), (NNlib.tanh_fast, _tanh),
    (NNlib.softplus, _softplus), (NNlib.softsign, _softsign), (NNlib.mish, _mish),
    (NNlib.lisht, _lisht), (NNlib.tanhshrink, _tanhshrink), (NNlib.logcosh, _logcosh),
    (NNlib.softshrink, _softshrink), (NNlib.trelu, _trelu),
)
for (f, impl) in UNARY
    @eval begin
        B.broadcasted(::typeof($f), x::_FloatTensor) = $impl(x)
        B.broadcasted(::typeof($f), s::EagerScalar{<:Union{Float32,Float64}}) =
            _scalar_result($impl(_scalar_tensor_value(s)))
    end
end
# Activations with a constant second parameter, e.g. `leakyrelu.(x, 0.2)`.
for (f, impl) in ((NNlib.leakyrelu, _leakyrelu), (NNlib.elu, _elu), (NNlib.celu, _celu),
                  (NNlib.softshrink, _softshrink), (NNlib.trelu, _trelu))
    @eval B.broadcasted(::typeof($f), x::_FloatTensor, p::Real) = $impl(x, p)
end

# Every function is usable as is; there is no faster tensor variant to swap in.
NNlib.fast_act(f::F, ::Union{_ADTensor,EagerScalar}) where {F<:Function} = f

# Softmax family -------------------------------------------------------------------------

NNlib.softmax(x::_FloatTensor; dims=1) = Tenferro._softmax(x, dims)
NNlib.logsoftmax(x::_FloatTensor; dims=1) = Tenferro._softmax(x, dims; log=true)
NNlib.logsumexp(x::_FloatTensor; dims=:) = Tenferro._logsumexp(x, dims)

# Batched matrix multiplication ---------------------------------------------------------

NNlib.batched_mul(a::_ADTensor, b::Union{_ADTensor,AbstractArray}) = Tenferro._batched_mul(a, b)
NNlib.batched_mul(a::AbstractArray, b::_ADTensor) = Tenferro._batched_mul(a, b)
NNlib.batched_transpose(a::_ADTensor) = Tenferro._batched_transpose(a)
NNlib.batched_adjoint(a::_ADTensor) = Tenferro._batched_adjoint(a)

# Padding --------------------------------------------------------------------------------

_pad_pairs(x, pad::Tuple{Vararg{Tuple{Int,Int}}}, dims) = pad
_pad_pairs(x, pad, dims) = NNlib.gen_pad(pad, dims isa Colon ? dims : (dims...,), Val(ndims(x)))

function NNlib.pad_constant(x::_ADTensor, pad::Union{Int,Tuple}, val=0; dims=:)
    pairs = _pad_pairs(x, pad, dims)
    length(pairs) == ndims(x) || throw(ArgumentError("padding must contain one (low, high) pair per dimension"))
    Tenferro._pad_constant(x, map(first, pairs), map(last, pairs), val)
end
NNlib.pad_zeros(x::_ADTensor, pad; dims=:) = NNlib.pad_constant(x, pad, 0; dims=dims)

end
