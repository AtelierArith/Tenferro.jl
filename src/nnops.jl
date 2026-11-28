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

# Sliding windows ----------------------------------------------------------------------
#
# tenferro-rs has no convolution or `reduce_window` primitive. Convolution and
# pooling gather every window with one axis gather per spatial dimension: dimension
# `d` of length `n` becomes the pair `(k, out)` of kernel offset and output
# position. Non-overlapping windows (stride == kernel) are a plain reshape. The gathers' derivatives scatter-add back, so gradients follow.

_zeros(n) = ntuple(_ -> 0, n)

# Pad `x` for the windows of `_windows`; also return the offset of the first
# window in every spatial dimension (nonzero where `lo` crops).
function _window_input(x::_ADTensor, kernel, stride, dilation, lo, out; interior=_zeros(length(kernel)), fill=0)
    S, N = length(kernel), ndims(x)
    all(>(0), out) || throw(DimensionMismatch("the windows do not fit into the padded input"))
    n = ntuple(d -> x.dims[d] + max(x.dims[d] - 1, 0) * interior[d], S)
    plo = map(l -> max(l, 0), lo)
    off = map(l -> max(-l, 0), lo)
    need = ntuple(d -> off[d] + (out[d] - 1) * stride[d] + (kernel[d] - 1) * dilation[d] + 1, S)
    phi = ntuple(d -> max(need[d] - n[d] - plo[d], 0), S)
    rest = _zeros(N - S)
    y = if iszero(fill)
        _pad(x, (plo..., rest...), (phi..., rest...), (interior..., rest...))
    else
        all(iszero, interior) || throw(ArgumentError("interior padding requires a zero fill value"))
        _pad_constant(x, (plo..., rest...), (phi..., rest...), fill)
    end
    y, off
end

# The entries of every window at kernel offset `k` (0-based): one strided slice.
function _offset_slice(y::_ADTensor, off, k, stride, dilation, out)
    S, N = length(out), ndims(y)
    start = ntuple(d -> d <= S ? off[d] + k[d] * dilation[d] : 0, N)
    values = UInt[start...,
                  ntuple(d -> d <= S ? start[d] + (out[d] - 1) * stride[d] + 1 : y.dims[d], N)...,
                  ntuple(d -> d <= S ? stride[d] : 1, N)...]
    _slice(y, values, ntuple(d -> d <= S ? out[d] : y.dims[d], N))
end

"""
    _windows(x, kernel, stride, dilation, lo, out; flip=false, interior=zeros, fill=0)

Gather the sliding windows of the first `S = length(kernel)` dimensions of `x`.
Dimension `d` is first dilated with `interior[d]` zeros between entries and padded
with `lo[d]` entries before it (negative `lo` crops) and as many entries after it
as the windows need, all equal to `fill`. Window `o` (0-based) along `d` reads
padded positions `o * stride[d] + k * dilation[d]` for `k` in `0:kernel[d]-1`,
reversed when `flip`. The result has dimensions
`(kernel[1], out[1], ..., kernel[S], out[S], size(x)[S+1:end]...)`.
"""
function _windows(x::_ADTensor, kernel, stride, dilation, lo, out; flip::Bool=false,
                  interior=_zeros(length(kernel)), fill=0)
    S = length(kernel)
    y, off = _window_input(x, kernel, stride, dilation, lo, out; interior=interior, fill=fill)
    for d in 1:S
        axis = 2d - 1
        K, O = kernel[d], out[d]
        idx = [off[d] + o * stride[d] + (flip ? K - 1 - k : k) * dilation[d] + 1 for k in 0:(K - 1), o in 0:(O - 1)]
        before, after = y.dims[1:(axis - 1)], y.dims[(axis + 1):end]
        if vec(idx) == 1:(K * O)
            # Non-overlapping windows: a reshape of (a prefix of) the axis.
            y = K * O == y.dims[axis] ? y : _axis_slice(y, axis, 1, K * O, 1)
        else
            # One strided slice per kernel offset, concatenated along the axis;
            # faster than an axis gather in tenferro-rs (forward and backward).
            parts = map(1:K) do k
                part = _axis_slice(y, axis, idx[k, 1], O, stride[d])
                reshape(part, (before..., 1, O, after...))
            end
            y = K == 1 ? only(parts) : cat(parts...; dims=axis)
        end
        y = reshape(y, (before..., K, O, after...))
    end
    y
end

# `n` entries of `axis` starting at the 1-based `start`, `step` apart.
function _axis_slice(y::_ADTensor, axis, start, n, step)
    N = ndims(y)
    values = UInt[ntuple(i -> i == axis ? start - 1 : 0, N)...,
                  ntuple(i -> i == axis ? start + (n - 1) * step : y.dims[i], N)...,
                  ntuple(i -> i == axis ? step : 1, N)...]
    _slice(y, values, ntuple(i -> i == axis ? n : y.dims[i], N))
end

_conv_out(n, k, s, d, lo, hi) = div(n + lo + hi - (k - 1) * d - 1, s) + 1

"""
    _conv(x, w, stride, lo, hi, dilation, groups, flip; interior=zeros)

Convolution with NNlib's WHCN layout: `x` is `(spatial..., C_in, batch)` and `w`
is `(kernel..., C_in ÷ groups, C_out)`. `flip=true` reverses the kernel (NNlib's
default convolution); `flip=false` is cross-correlation. `interior` dilates the
input (transposed convolution).
"""
function _conv(x::_ADTensor, w::_ADTensor, stride, lo, hi, dilation, groups::Integer, flip::Bool;
               interior=_zeros(ndims(x) - 2))
    S = ndims(x) - 2
    ndims(w) == ndims(x) || throw(DimensionMismatch("input and kernel ranks must match"))
    kernel = w.dims[1:S]
    cin, batch, cout = x.dims[S + 1], x.dims[S + 2], w.dims[S + 2]
    g = Int(groups)
    cin % g == 0 && cout % g == 0 || throw(DimensionMismatch("channels must be divisible by groups"))
    w.dims[S + 1] * g == cin || throw(DimensionMismatch("expected $(w.dims[S + 1] * g) input channels, got $cin"))
    n = ntuple(d -> x.dims[d] + max(x.dims[d] - 1, 0) * interior[d], S)
    out = ntuple(d -> _conv_out(n[d], kernel[d], stride[d], dilation[d], lo[d], hi[d]), S)
    patches = _windows(x, kernel, stride, dilation, lo, out; flip=flip, interior=interior)
    kd = ntuple(d -> 2d - 1, S)
    y = if g == 1
        _dot_general(patches, w, (kd..., 2S + 1), (ntuple(identity, S)..., S + 1))
    else
        p = reshape(patches, (patches.dims[1:(2S)]..., cin ÷ g, g, batch))
        wg = reshape(w, (kernel..., cin ÷ g, cout ÷ g, g))
        reshape(_dot_general(p, wg, (kd..., 2S + 1), (ntuple(identity, S)..., S + 1), (2S + 2,), (S + 3,)),
                (out..., batch, cout))
    end
    permutedims(y, (ntuple(identity, S)..., S + 2, S + 1))
end
_conv(x::_ADTensor, w::AbstractArray, args...; kw...) = _conv(x, _operand(x, w), args...; kw...)
_conv(x::AbstractArray, w::_ADTensor, args...; kw...) = _conv(_operand(w, x), w, args...; kw...)

# Swap the channel dimensions of a grouped kernel: `(kernel..., C_in ÷ g, C_out)`
# becomes `(kernel..., C_out ÷ g, C_in)`.
function _transpose_kernel(w::_ADTensor, groups)
    S = ndims(w) - 2
    g = Int(groups)
    kernel, ci, co = w.dims[1:S], w.dims[S + 1], w.dims[S + 2]
    sp = ntuple(identity, S)
    g == 1 && return permutedims(w, (sp..., S + 2, S + 1))
    wt = permutedims(reshape(w, (kernel..., ci, co ÷ g, g)), (sp..., S + 2, S + 1, S + 3))
    reshape(wt, (kernel..., co ÷ g, ci * g))
end

"""
    _conv_data(dy, w, insize, stride, lo, hi, dilation, groups, flip)

Gradient of `_conv` with respect to its input of spatial size `insize`: the
transposed convolution of `dy`, i.e. a stride-one convolution of the
interior-padded `dy` with the channel-swapped kernel reversed once more.
"""
function _conv_data(dy::_ADTensor, w::_ADTensor, insize, stride, lo, hi, dilation, groups, flip::Bool)
    S = ndims(dy) - 2
    kernel = w.dims[1:S]
    out = dy.dims[1:S]
    lo2 = ntuple(d -> (kernel[d] - 1) * dilation[d] - lo[d], S)
    hi2 = ntuple(d -> insize[d] - (out[d] - 1) * stride[d] - 1 + lo[d], S)
    _conv(dy, _transpose_kernel(w, groups), ntuple(_ -> 1, S), lo2, hi2, dilation, groups, !flip;
          interior=map(s -> s - 1, stride))
end
_conv_data(dy::_ADTensor, w::AbstractArray, args...) = _conv_data(dy, _operand(dy, w), args...)
_conv_data(dy::AbstractArray, w::_ADTensor, args...) = _conv_data(_operand(w, dy), w, args...)

"""
    _conv_filter(x, dy, kernel, stride, lo, dilation, groups, flip)

Gradient of `_conv` with respect to its kernel of spatial size `kernel`.
"""
function _conv_filter(x::_ADTensor, dy::_ADTensor, kernel, stride, lo, dilation, groups, flip::Bool)
    S = ndims(x) - 2
    out = dy.dims[1:S]
    cin, batch, cout = x.dims[S + 1], x.dims[S + 2], dy.dims[S + 1]
    g = Int(groups)
    patches = _windows(x, kernel, stride, dilation, lo, out; flip=flip)
    od = ntuple(d -> 2d, S)
    sp = ntuple(identity, S)
    g == 1 && return _dot_general(patches, dy, (od..., 2S + 2), (sp..., S + 2))
    p = reshape(patches, (patches.dims[1:(2S)]..., cin ÷ g, g, batch))
    dyg = reshape(dy, (out..., cout ÷ g, g, batch))
    reshape(_dot_general(p, dyg, (od..., 2S + 3), (sp..., S + 3), (2S + 2,), (S + 2,)), (kernel..., cin ÷ g, cout))
end
_conv_filter(x::_ADTensor, dy::AbstractArray, args...) = _conv_filter(x, _operand(x, dy), args...)
_conv_filter(x::AbstractArray, dy::_ADTensor, args...) = _conv_filter(_operand(dy, x), dy, args...)

# Pooling -------------------------------------------------------------------------------

# Number of window positions `o * stride - lo + k * dilation` (0-based, k < kernel)
# that are at least `lower` and below `upper`.
function _window_count(o, kernel, stride, dilation, lo, lower, upper)
    start = o * stride - lo
    count(k -> lower <= start + k * dilation < upper, 0:(kernel - 1))
end

"""
    _pool(x, mode, kernel, stride, lo, hi, dilation, out; p=2, count_include_pad=true)

Max (`mode = :max`), mean (`:mean`) or Lp-norm (`:lpnorm`) pooling over the
spatial dimensions of `x` with NNlib's conventions. Max pooling scans the
kernel offsets like NNlib, replacing the running maximum only by larger entries,
so each window's gradient goes to its first maximum, as with `NNlib.∇maxpool`;
mean pooling divides by the number of window positions inside the padded input
(or inside the input when `count_include_pad=false`).
"""
function _pool(x::_ADTensor, mode::Symbol, kernel, stride, lo, hi, dilation, out; p=2, count_include_pad::Bool=true)
    S = length(kernel)
    T = eltype(x)
    rest = x.dims[(S + 1):end]
    kd = ntuple(d -> 2d - 1, S)
    # Non-overlapping windows are a reshape: reduce them directly. Otherwise
    # combine one output-sized slice per kernel offset.
    disjoint = all(d -> stride[d] == kernel[d] && (dilation[d] == 1 || kernel[d] == 1), 1:S)
    if mode === :max
        # NNlib's scan: a later entry replaces the running maximum only if it is
        # larger, so the gradient of each window goes to its first maximum.
        y, off = _window_input(x, kernel, stride, dilation, lo, out; fill=typemin(T))
        m = nothing
        for k in CartesianIndices(kernel)
            v = _offset_slice(y, off, Tuple(k) .- 1, stride, dilation, out)
            m = m === nothing ? v : _select(_B.broadcasted(>, v, m), v, m)
        end
        return m
    end
    lift = mode === :lpnorm ? (v -> _B.broadcasted(^, v, p)) : identity
    s = if disjoint
        reshape(sum(lift(_windows(x, kernel, stride, dilation, lo, out)); dims=kd), (out..., rest...))
    else
        y, off = _window_input(x, kernel, stride, dilation, lo, out)
        acc = nothing
        for k in CartesianIndices(kernel)
            v = lift(_offset_slice(y, off, Tuple(k) .- 1, stride, dilation, out))
            acc = acc === nothing ? v : _B.broadcasted(+, acc, v)
        end
        acc
    end
    if mode === :mean
        counts = ntuple(S) do d
            upper = count_include_pad ? x.dims[d] + hi[d] : x.dims[d]
            lower = count_include_pad ? -lo[d] : 0
            [_window_count(o, kernel[d], stride[d], dilation[d], lo[d], lower, upper) for o in 0:(out[d] - 1)]
        end
        divisor = ones(T, (out..., map(_ -> 1, rest)...))
        for I in CartesianIndices(out)
            c = prod(d -> counts[d][I[d]], 1:S)
            divisor[I] = c == 0 ? floatmax(T) : c
        end
        return _B.broadcasted(/, s, divisor)
    elseif mode === :lpnorm
        return _B.broadcasted(^, s, inv(convert(T, p)))
    end
    throw(ArgumentError("unknown pooling mode $mode"))
end

# Upsampling -----------------------------------------------------------------------------

"""
    _upsample_nearest(x, scales)

Repeat every entry of the first `length(scales)` dimensions `scales[d]` times.
"""
function _upsample_nearest(x::_ADTensor, scales)
    S = length(scales)
    all(isone, scales) && return x
    sz = x.dims
    expanded = ntuple(i -> i <= 2S ? (isodd(i) ? 1 : sz[i ÷ 2]) : sz[i - S], ndims(x) + S)
    target = ntuple(i -> i <= 2S ? (isodd(i) ? scales[(i + 1) ÷ 2] : sz[i ÷ 2]) : sz[i - S], ndims(x) + S)
    y = _broadcast_to(reshape(x, expanded), target)
    reshape(y, ntuple(i -> i <= S ? sz[i] * scales[i] : sz[i], ndims(x)))
end

# NNlib's source positions and weights for linear interpolation of an axis of
# length `n_in` onto `n_out` points.
function _linear_sources(::Type{T}, n_in, n_out, align_corners) where {T}
    ratio = align_corners ? (n_out > 1 ? T((n_in - 1) / (n_out - 1)) : zero(T)) : T(n_in / n_out)
    i0, i1, λ0, λ1 = Int[], Int[], T[], T[]
    for o in 0:(n_out - 1)
        real_index = align_corners ? ratio * o : max(zero(T), ratio * (o + T(0.5)) - T(0.5))
        j = floor(Int, real_index)
        push!(i0, j + 1)
        push!(i1, j + 1 + (j < n_in - 1 ? 1 : 0))
        w1 = real_index - j
        push!(λ1, w1)
        push!(λ0, one(T) - w1)
    end
    i0, i1, λ0, λ1
end

"""
    _upsample_linear(x, size; align_corners=true)

Separable (bi/tri)linear interpolation of the leading dimensions of `x` to `size`.
"""
function _upsample_linear(x::_ADTensor, size; align_corners::Bool=true)
    T = eltype(x)
    y = x
    for d in eachindex(size)
        n_in, n_out = x.dims[d], size[d]
        n_in == n_out && continue
        i0, i1, λ0, λ1 = _linear_sources(T, n_in, n_out, align_corners)
        shape = ntuple(i -> i == d ? n_out : 1, ndims(x))
        a = _B.broadcasted(*, _gather_axis(y, i0, d), reshape(λ0, shape))
        b = _B.broadcasted(*, _gather_axis(y, i1, d), reshape(λ1, shape))
        y = _B.broadcasted(+, a, b)
    end
    y
end

function _pixel_shuffle(x::_ADTensor, r::Integer)
    ndims(x) > 2 || throw(ArgumentError("expected x with at least 3 dimensions"))
    d = ndims(x) - 2
    sizein = x.dims[1:d]
    cin, n = x.dims[d + 1], x.dims[d + 2]
    cin % r^d == 0 || throw(ArgumentError("expected channel dimension to be divisible by r^d = $(r^d), " *
                                          "where d=$d is the number of spatial dimensions. Given r=$r, input size(x) = $(x.dims)"))
    cout = cin ÷ r^d
    y = reshape(x, (sizein..., ntuple(_ -> Int(r), d)..., cout, n))
    perm = ntuple(i -> isodd(i) ? d + (i + 1) ÷ 2 : i ÷ 2, 2d)
    y = permutedims(y, (perm..., 2d + 1, 2d + 2))
    reshape(y, (map(s -> s * Int(r), sizein)..., cout, n))
end
