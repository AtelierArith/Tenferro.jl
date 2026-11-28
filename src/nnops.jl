# Neural-network building blocks shared by the NNlib extension and plain users:
# broadcast fallback, activation kernels, softmax, zero padding and dot_general.

const _B = Base.Broadcast
const _FloatTensor = Union{EagerTensor{<:Union{Float32,Float64}},TracedTensor{<:Union{Float32,Float64}}}

# Broadcast fallback -------------------------------------------------------------

# Supported broadcasts are lowered op by op through specific `broadcasted`
# methods. The style only routes every other function to a clear error instead
# of Julia's generic broadcast machinery, which cannot iterate tensors.
struct _TensorStyle <: _B.BroadcastStyle end
_B.broadcastable(x::Union{_ADTensor,EagerScalar}) = x
_B.BroadcastStyle(::Type{<:Union{_ADTensor,EagerScalar}}) = _TensorStyle()
_B.BroadcastStyle(::_TensorStyle, ::_B.BroadcastStyle) = _TensorStyle()
_B.BroadcastStyle(::_TensorStyle, ::_B.Unknown) = _TensorStyle()

# `(f ∘ g).(x...)` is `f.(g.(x...))`; LuxLib broadcasts `σ ∘ +`, for example.
_B.broadcasted(::_TensorStyle, f::ComposedFunction, args...) = _B.broadcasted(f.outer, _B.broadcasted(f.inner, args...))
function _B.broadcasted(::_TensorStyle, f::F, args...) where {F}
    throw(ArgumentError("broadcasting `$f` over Tenferro tensors is not supported. Only the elementwise " *
                        "functions listed in the Tenferro documentation are lowered to tenferro-rs; " *
                        "write an anonymous function or closure as separate dotted operations."))
end
_B.broadcasted(::typeof(identity), a::Union{_ADTensor,EagerScalar}) = a

# Activation kernels -------------------------------------------------------------

function _activation(a::_FloatTensor, code)
    id = _bridge(a, RustCrate.activation, RustCrate.traced_activation, UInt(a.id), Int32(code))
    _wrap(a, eltype(a), id, a.dims)
end
_sigmoid(a::_FloatTensor) = _activation(a, 0)
_silu(a::_FloatTensor) = _activation(a, 1)
_softplus(a::_FloatTensor) = _activation(a, 2)
_gelu_erf(a::_FloatTensor) = _activation(a, 3)
_gelu_tanh(a::_FloatTensor) = _activation(a, 4)

# Softmax family ------------------------------------------------------------------

function _softmax_axis(a::_FloatTensor, axis::Integer, log::Bool)
    id = _bridge(a, RustCrate.softmax_node, RustCrate.traced_softmax_node, UInt(a.id), UInt(axis - 1), log)
    _wrap(a, eltype(a), id, a.dims)
end

# The shift by the maximum keeps `exp` finite; its derivative cancels exactly,
# so it needs no stop-gradient.
function _shifted(a::_FloatTensor, dims)
    m = maximum(a; dims=dims)
    shifted = _B.broadcasted(-, a, m)
    m, shifted
end

function _softmax(a::_FloatTensor, dims; log::Bool=false)
    if dims isa Integer && 1 <= dims <= ndims(a) && a.dims[dims] > 0
        return _softmax_axis(a, dims, log)
    end
    _, shifted = _shifted(a, dims)
    e = _B.broadcasted(exp, shifted)
    s = sum(e; dims=dims)
    log ? _B.broadcasted(-, shifted, _B.broadcasted(Base.log, s)) : _B.broadcasted(/, e, s)
end
_B.broadcasted(::typeof(log), s::EagerScalar) = _scalar_result(_B.broadcasted(log, _scalar_tensor_value(s)))
_B.broadcasted(::typeof(exp), s::EagerScalar) = _scalar_result(_B.broadcasted(exp, _scalar_tensor_value(s)))

function _logsumexp(a::_FloatTensor, dims)
    if dims isa Colon
        y = _logsumexp(reshape(a, length(a)), 1)
        return y[1]
    end
    m, shifted = _shifted(a, dims)
    _B.broadcasted(+, m, _B.broadcasted(log, sum(_B.broadcasted(exp, shifted); dims=dims)))
end

# Padding --------------------------------------------------------------------------

"""
    _pad(a, low, high, interior=zeros)

Zero-pad `a` with `low[d]`/`high[d]` entries before/after dimension `d` and
`interior[d]` zeros between neighbouring entries (StableHLO `pad`).
"""
function _pad(a::_ADTensor, low, high, interior=ntuple(_ -> 0, ndims(a)))
    N = ndims(a)
    length(low) == length(high) == length(interior) == N || throw(ArgumentError("one padding amount per dimension is required"))
    all(>=(0), interior) || throw(ArgumentError("interior padding must be nonnegative"))
    dims = ntuple(N) do d
        n = a.dims[d]
        n + low[d] + high[d] + max(n - 1, 0) * interior[d]
    end
    all(>=(0), dims) || throw(ArgumentError("padding removes more entries than the tensor has"))
    all(iszero, low) && all(iszero, high) && all(iszero, interior) && return a
    amounts = Int64[low..., high..., interior...]
    id = _bridge(a, RustCrate.pad_node, RustCrate.traced_pad_node, UInt(a.id), amounts)
    _wrap(a, eltype(a), id, dims)
end

# Pad with a constant `val`: zero padding plus a constant frame.
function _pad_constant(a::_ADTensor, low, high, val)
    padded = _pad(a, low, high)
    iszero(val) && return padded
    T = eltype(a)
    frame = fill(convert(T, val), padded.dims)
    frame[ntuple(d -> (low[d] + 1):(low[d] + a.dims[d]), ndims(a))...] .= zero(T)
    _B.broadcasted(+, padded, _constant_tensor(a, frame))
end

# Generalized contraction -------------------------------------------------------------

_operand(ref::_ADTensor, x::_ADTensor) = (_same_graph(ref, x); x)
_operand(ref::_ADTensor, x::AbstractArray) = _constant_tensor(ref, x)

"""
    _dot_general(a, b, lhs_contract, rhs_contract, lhs_batch=(), rhs_batch=())

Contract `a` and `b` over the given (1-based) dimensions. The result has the
free dimensions of `a`, then the free dimensions of `b`, then the batch
dimensions, matching tenferro-rs `dot_general`.
"""
function _dot_general(a::_ADTensor, b::_ADTensor, lc, rc, lb=(), rb=())
    _same_graph(a, b)
    T = promote_type(eltype(a), eltype(b))
    T <: ADElement || throw(ArgumentError("unsupported promoted tensor element type: $T"))
    a, b = _convert_tensor(a, T), _convert_tensor(b, T)
    length(lc) == length(rc) || throw(ArgumentError("contracting dimension lists must have equal length"))
    length(lb) == length(rb) || throw(ArgumentError("batch dimension lists must have equal length"))
    all(i -> a.dims[lc[i]] == b.dims[rc[i]], eachindex(lc)) || throw(DimensionMismatch("contracting dimensions do not match"))
    all(i -> a.dims[lb[i]] == b.dims[rb[i]], eachindex(lb)) || throw(DimensionMismatch("batch dimensions do not match"))
    afree = Tuple(d for d in 1:ndims(a) if !(d in lc) && !(d in lb))
    bfree = Tuple(d for d in 1:ndims(b) if !(d in rc) && !(d in rb))
    dims = (map(d -> a.dims[d], afree)..., map(d -> b.dims[d], bfree)..., map(d -> a.dims[d], Tuple(lb))...)
    spec = UInt[length(lc), length(lb), (lc .- 1)..., (rc .- 1)..., (lb .- 1)..., (rb .- 1)...]
    id = _bridge(a, RustCrate.dot_general_node, RustCrate.traced_dot_general_node, UInt(a.id), UInt(b.id), spec)
    _wrap(a, T, id, dims)
end

# Batched matrix multiplication with NNlib's conventions: batch dimensions
# trail, and a 3-D batch of size one is broadcast.
function _batched_mul(a::_ADTensor, b::_ADTensor)
    if ndims(a) == 2 && ndims(b) == 3
        return _dot_general(a, b, (2,), (1,))
    elseif ndims(a) == 3 && ndims(b) == 2
        return permutedims(_dot_general(a, b, (2,), (1,)), (1, 3, 2))
    end
    ndims(a) == ndims(b) >= 3 || throw(DimensionMismatch("batched_mul requires matching ranks of at least 3, or a matrix and a 3-D tensor"))
    if ndims(a) > 3
        size(a)[3:end] == size(b)[3:end] || throw(DimensionMismatch("batch size has to be the same for the two arrays"))
        batch = size(a)[3:end]
        c = _batched_mul(reshape(a, size(a, 1), size(a, 2), :), reshape(b, size(b, 1), size(b, 2), :))
        return reshape(c, size(c, 1), size(c, 2), batch...)
    end
    na, nb = size(a, 3), size(b, 3)
    na == nb || na == 1 || nb == 1 || throw(DimensionMismatch("batch size mismatch: A != B"))
    size(a, 2) == size(b, 1) || throw(DimensionMismatch("matrix dimensions do not match"))
    if na == 1 && nb != 1
        return _dot_general(reshape(a, size(a, 1), size(a, 2)), b, (2,), (1,))
    elseif nb == 1 && na != 1
        return permutedims(_dot_general(a, reshape(b, size(b, 1), size(b, 2)), (2,), (1,)), (1, 3, 2))
    end
    _dot_general(a, b, (2,), (1,), (3,), (3,))
end
_batched_mul(a::_ADTensor, b::AbstractArray) = _batched_mul(a, _operand(a, b))
_batched_mul(a::AbstractArray, b::_ADTensor) = _batched_mul(_operand(b, a), b)

function _batched_transpose(a::_ADTensor)
    ndims(a) >= 2 || throw(ArgumentError("batched_transpose requires at least two dimensions"))
    permutedims(a, (2, 1, ntuple(i -> i + 2, ndims(a) - 2)...))
end
_batched_adjoint(a::_ADTensor) = eltype(a) <: Real ? _batched_transpose(a) : conj(_batched_transpose(a))
