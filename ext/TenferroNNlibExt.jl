module TenferroNNlibExt

using NNlib
using Tenferro: Tenferro, Statistics, EagerScalar, _ADTensor, _FloatTensor, _select, _scalar_result, _scalar_tensor_value

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

_relu(x) = Tenferro._relu(x)
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

# `relu` dense layers run on the native dense kernel (`Tenferro.dense`).
Tenferro._dense_act(::typeof(NNlib.relu)) = Int32(1)

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

# Convolution ------------------------------------------------------------------------------

const _Arg = Union{_ADTensor,AbstractArray}
const _P = Tenferro

_lo(c) = ntuple(d -> NNlib.padding(c)[2d - 1], NNlib.spatial_dims(c))
_hi(c) = ntuple(d -> NNlib.padding(c)[2d], NNlib.spatial_dims(c))

_conv_cdims(x, w, cdims::DenseConvDims) =
    _P._conv(x, w, NNlib.stride(cdims), _lo(cdims), _hi(cdims), NNlib.dilation(cdims),
             NNlib.groupcount(cdims), !NNlib.flipkernel(cdims))
NNlib.conv(x::_ADTensor, w::_Arg, cdims::DenseConvDims; kwargs...) = _conv_cdims(x, w, cdims)
NNlib.conv(x::AbstractArray, w::_ADTensor, cdims::DenseConvDims; kwargs...) = _conv_cdims(x, w, cdims)

function _conv_kw(x, w; stride=1, pad=0, dilation=1, flipped=false, groups=1)
    S = ndims(w) - 2
    cdims = DenseConvDims(size(x), size(w); stride=NNlib.expand(Val(S), stride), padding=NNlib.expand(Val(S), pad),
                          dilation=NNlib.expand(Val(S), dilation), flipkernel=flipped, groups=groups)
    _conv_cdims(x, w, cdims)
end
NNlib.conv(x::_ADTensor, w::_ADTensor; kwargs...) = _conv_kw(x, w; kwargs...)
NNlib.conv(x::_ADTensor, w::AbstractArray; kwargs...) = _conv_kw(x, w; kwargs...)
NNlib.conv(x::AbstractArray, w::_ADTensor; kwargs...) = _conv_kw(x, w; kwargs...)

function _depthwise_kw(x, w; stride=1, pad=0, dilation=1, flipped=false)
    S = ndims(w) - 2
    cdims = DepthwiseConvDims(size(x), size(w); stride=NNlib.expand(Val(S), stride), padding=NNlib.expand(Val(S), pad),
                              dilation=NNlib.expand(Val(S), dilation), flipkernel=flipped)
    NNlib.depthwiseconv(x, w, cdims)
end
NNlib.depthwiseconv(x::_ADTensor, w::_ADTensor; kwargs...) = _depthwise_kw(x, w; kwargs...)
NNlib.depthwiseconv(x::_ADTensor, w::AbstractArray; kwargs...) = _depthwise_kw(x, w; kwargs...)
NNlib.depthwiseconv(x::AbstractArray, w::_ADTensor; kwargs...) = _depthwise_kw(x, w; kwargs...)

_conv_data_cdims(dy, w, cdims::DenseConvDims) =
    _P._conv_data(dy, w, NNlib.input_size(cdims), NNlib.stride(cdims), _lo(cdims), _hi(cdims),
                  NNlib.dilation(cdims), NNlib.groupcount(cdims), !NNlib.flipkernel(cdims))
NNlib.∇conv_data(dy::_ADTensor, w::_Arg, cdims::DenseConvDims; kwargs...) = _conv_data_cdims(dy, w, cdims)
NNlib.∇conv_data(dy::AbstractArray, w::_ADTensor, cdims::DenseConvDims; kwargs...) = _conv_data_cdims(dy, w, cdims)

_conv_filter_cdims(x, dy, cdims::DenseConvDims) =
    _P._conv_filter(x, dy, NNlib.kernel_size(cdims), NNlib.stride(cdims), _lo(cdims), NNlib.dilation(cdims),
                    NNlib.groupcount(cdims), !NNlib.flipkernel(cdims))
NNlib.∇conv_filter(x::_ADTensor, dy::_Arg, cdims::DenseConvDims; kwargs...) = _conv_filter_cdims(x, dy, cdims)
NNlib.∇conv_filter(x::AbstractArray, dy::_ADTensor, cdims::DenseConvDims; kwargs...) = _conv_filter_cdims(x, dy, cdims)

# Depthwise kernels `(kernel..., C_mult, C_in)` are grouped kernels `(kernel..., 1, C_mult * C_in)`.
_grouped(w) = reshape(w, (size(w)[1:(end - 2)]..., 1, size(w, ndims(w) - 1) * size(w, ndims(w))))
NNlib.depthwiseconv(x::_ADTensor, w::_Arg, cdims::DepthwiseConvDims; kwargs...) = _conv_cdims(x, _grouped(w), DenseConvDims(cdims))
NNlib.depthwiseconv(x::AbstractArray, w::_ADTensor, cdims::DepthwiseConvDims; kwargs...) =
    _conv_cdims(x, _grouped(w), DenseConvDims(cdims))
NNlib.∇depthwiseconv_data(dy::_ADTensor, w::_Arg, cdims::DepthwiseConvDims; kwargs...) =
    _conv_data_cdims(dy, _grouped(w), DenseConvDims(cdims))
NNlib.∇depthwiseconv_data(dy::AbstractArray, w::_ADTensor, cdims::DepthwiseConvDims; kwargs...) =
    _conv_data_cdims(dy, _grouped(w), DenseConvDims(cdims))
function _depthwise_filter(x, dy, cdims::DepthwiseConvDims)
    dw = _conv_filter_cdims(x, dy, DenseConvDims(cdims))
    reshape(dw, (NNlib.kernel_size(cdims)..., NNlib.channel_multiplier(cdims), NNlib.channels_in(cdims)))
end
NNlib.∇depthwiseconv_filter(x::_ADTensor, dy::_Arg, cdims::DepthwiseConvDims; kwargs...) = _depthwise_filter(x, dy, cdims)
NNlib.∇depthwiseconv_filter(x::AbstractArray, dy::_ADTensor, cdims::DepthwiseConvDims; kwargs...) = _depthwise_filter(x, dy, cdims)

NNlib.DenseConvDims(x::_ADTensor, w::_Arg; kwargs...) = DenseConvDims(size(x), size(w); kwargs...)
NNlib.DenseConvDims(x::AbstractArray, w::_ADTensor; kwargs...) = DenseConvDims(size(x), size(w); kwargs...)
NNlib.DepthwiseConvDims(x::_ADTensor, w::_Arg; kwargs...) = DepthwiseConvDims(size(x), size(w); kwargs...)
NNlib.DepthwiseConvDims(x::AbstractArray, w::_ADTensor; kwargs...) = DepthwiseConvDims(size(x), size(w); kwargs...)

# Pooling -----------------------------------------------------------------------------------

NNlib.PoolDims(x::_ADTensor, k; kwargs...) = PoolDims(size(x), k; kwargs...)

_pool(x, mode, pdims::PoolDims; kwargs...) =
    _P._pool(x, mode, NNlib.kernel_size(pdims), NNlib.stride(pdims), _lo(pdims), _hi(pdims),
             NNlib.dilation(pdims), NNlib.output_size(pdims); kwargs...)
NNlib.maxpool(x::_ADTensor, pdims::PoolDims; kwargs...) = _pool(x, :max, pdims)
NNlib.meanpool(x::_ADTensor, pdims::PoolDims; count_include_pad::Bool=true, kwargs...) =
    _pool(x, :mean, pdims; count_include_pad=count_include_pad)
function NNlib.lpnormpool(x::_ADTensor, pdims::PoolDims; p, kwargs...)
    (isinf(p) || p < 0) && error("p value of Lp norm pool expects `0 < p < Inf`, but p is $(p) now.")
    _pool(x, :lpnorm, pdims; p=p)
end

# Upsampling ---------------------------------------------------------------------------------

NNlib.upsample_nearest(x::_ADTensor, scales::NTuple{S,Integer}) where {S} =
    (S in 1:ndims(x) || throw(ArgumentError("can't upsample ndims(x)=$(ndims(x)) with scale=$scales"));
     _P._upsample_nearest(x, scales))
function NNlib.upsample_nearest(x::_ADTensor; size::NTuple{S,Int}) where {S}
    xsize = Base.size(x)[1:S]
    all(size .% xsize .== 0) || throw(ArgumentError("expected output size divisible by input size"))
    _P._upsample_nearest(x, size .÷ xsize)
end

function NNlib.upsample_linear(x::_ADTensor, scale::NTuple{M,Real}; align_corners::Bool=true) where {M}
    N = ndims(x)
    M == N - 2 || error("The scale argument should be an NTuple with length $(N - 2), but it has length $M.")
    NNlib.upsample_linear(x; size=ntuple(i -> floor(Int, scale[i] * Base.size(x, i)), N - 2), align_corners)
end
NNlib.upsample_linear(x::_ADTensor, scale::Real; align_corners::Bool=true) =
    NNlib.upsample_linear(x, ntuple(_ -> scale, ndims(x) - 2); align_corners)
function NNlib.upsample_linear(x::_ADTensor; size::Union{Integer,NTuple{<:Any,Integer}}, align_corners::Bool=true)
    length(size) == ndims(x) - 2 ||
        error("The scale argument should be an NTuple with length $(ndims(x) - 2), but it has length $(length(size)).")
    _P._upsample_linear(x, Tuple(size); align_corners)
end

NNlib.pixel_shuffle(x::_ADTensor, r::Integer) = _P._pixel_shuffle(x, r)

# Gather and scatter -----------------------------------------------------------------------

NNlib.gather(src::_ADTensor, idx::AbstractArray; checkbounds::Bool=true) = _P._gather(src, idx)
NNlib.gather(src::_ADTensor, I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer},
             Ks::AbstractVector{<:Integer}...; checkbounds::Bool=true) = _P._gather(src, CartesianIndex.(I, J, Ks...))

for (op, name) in ((+, :+), (-, :-), (Statistics.mean, :mean))
    @eval NNlib.scatter(::typeof($op), src::_ADTensor, idx::AbstractArray; init=nothing, dstsize=nothing,
                        checkbounds::Bool=true) = _P._scatter($(QuoteNode(name)), src, idx, init, dstsize)
end
NNlib.scatter(op, src::_ADTensor, idx::AbstractArray; kwargs...) =
    throw(ArgumentError("NNlib.scatter on Tenferro tensors supports +, - and mean, not $op"))

end
