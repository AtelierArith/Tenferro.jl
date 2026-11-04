function Base.Array(x::EagerTensor{T}) where {T}
    out = Array{T}(undef, x.dims)
    _with_output_buffer(out) do data, len
        RustCall.result_to_exception(_rust_call(x.graph, RustCrate.copy_array, UInt(x.id), data, len, _typed_kind(T)))
    end
    return out
end

for (op, code) in ((:+, 0), (:-, 1))
    @eval function Base.$op(a::EagerTensor{T,N}, b::EagerTensor{T,N}) where {T,N}
        a.graph === b.graph || throw(ArgumentError("tensors belong to different runtimes"))
        a.dims == b.dims || throw(DimensionMismatch("tensor shapes do not match"))
        id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.binary, UInt(a.id), UInt(b.id), Int32($code)))
        id >= 0 || error("tenferro-rs operation failed")
        EagerTensor{T,N,typeof(a.graph)}(a.graph, id, a.dims)
    end
end

function Base.:*(a::EagerTensor{T,2}, b::EagerTensor{T,2}) where {T}
    a.graph === b.graph || throw(ArgumentError("tensors belong to different runtimes"))
    a.dims[2] == b.dims[1] || throw(DimensionMismatch("matrix dimensions do not match"))
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.matmul, UInt(a.id), UInt(b.id)))
    id >= 0 || error("tenferro-rs matrix multiplication failed")
    EagerTensor{T,2,typeof(a.graph)}(a.graph, id, (a.dims[1], b.dims[2]))
end

function Base.sum(a::EagerTensor{T}; dims=:) where {T}
    dims isa Colon || return _reduce_dims(a, dims, 2)
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.sum, UInt(a.id)))
    id >= 0 || error("tenferro-rs reduction failed")
    EagerScalar{T,typeof(a.graph)}(a.graph, id)
end

function Base.Array(x::TracedTensor{T}) where {T}
    out = Array{T}(undef, x.dims)
    _with_output_buffer(out) do data, len
        RustCall.result_to_exception(_rust_call(x.graph, RustCrate.traced_copy_array, UInt(x.id), data, len, _typed_kind(T)))
    end
    return out
end

for (op, code) in ((:+, 0), (:-, 1))
    @eval function Base.$op(a::TracedTensor{T,N}, b::TracedTensor{T,N}) where {T,N}
        a.graph === b.graph || throw(ArgumentError("tensors belong to different contexts"))
        a.dims == b.dims || throw(DimensionMismatch("tensor shapes do not match"))
        id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_binary, UInt(a.id), UInt(b.id), Int32($code)))
        id >= 0 || error("tenferro-rs traced operation failed")
        return TracedTensor{T,N,typeof(a.graph)}(a.graph, id, a.dims)
    end
end

function Base.:*(a::TracedTensor{T,2}, b::TracedTensor{T,2}) where {T}
    a.graph === b.graph || throw(ArgumentError("tensors belong to different contexts"))
    a.dims[2] == b.dims[1] || throw(DimensionMismatch("matrix dimensions do not match"))
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_binary, UInt(a.id), UInt(b.id), Int32(2)))
    id >= 0 || error("tenferro-rs traced matrix multiplication failed")
    return TracedTensor{T,2,typeof(a.graph)}(a.graph, id, (a.dims[1], b.dims[2]))
end

function Base.sum(a::TracedTensor{T}; dims=:) where {T}
    dims isa Colon || return _reduce_dims(a, dims, 2)
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_sum, UInt(a.id)))
    id >= 0 || error("tenferro-rs traced reduction failed")
    return TracedTensor{T,0,typeof(a.graph)}(a.graph, id, ())
end

Base.size(x::TypedTensor) = x.dims
Base.IndexStyle(::Type{<:TypedTensor}) = IndexLinear()
Base.similar(::TypedTensor, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)

function Base.Array(x::TypedTensor{T,N}) where {T,N}
    out = Array{T}(undef, x.dims)
    _with_output_buffer(out) do data, len
        _rust_call(x.storage, RustCrate.typed_copy, data, len) == 0 || error("tenferro-rs could not read typed tensor")
    end
    return out
end

function Base.getindex(x::TypedTensor{T}, i::Int) where {T}
    @boundscheck checkbounds(x, i)
    out = Ref{T}()
    _with_scalar_buffer(out) do data
        _rust_call(x.storage, RustCrate.typed_get, UInt(i - 1), data) == 0 || throw(BoundsError(x, i))
    end
    return out[]
end

function Base.setindex!(x::TypedTensor{T}, value, i::Int) where {T}
    @boundscheck checkbounds(x, i)
    data = Ref{T}(convert(T, value))
    _with_scalar_buffer(data) do ptr
        _rust_call(x.storage, RustCrate.typed_set, UInt(i - 1), ptr) == 0 || throw(BoundsError(x, i))
    end
    return x
end

# Lower each supported broadcast operation directly into the Rust graph.
const _ADTensor = Union{EagerTensor,TracedTensor}

for (op, code) in ((:-, 0), (:exp, 1), (:log, 2), (:sqrt, 3), (:sin, 4), (:cos, 5), (:tanh, 6), (:expm1, 7), (:log1p, 8), (:conj, 9), (:sign, 10))
    @eval begin
        function _unary(::typeof($op), a::EagerTensor{T,N}) where {T,N}
            id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.unary, UInt(a.id), Int32($code)))
            return EagerTensor{T,N,typeof(a.graph)}(a.graph, id, a.dims)
        end
        function _unary(::typeof($op), a::TracedTensor{T,N}) where {T,N}
            id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_unary, UInt(a.id), Int32($code)))
            return TracedTensor{T,N,typeof(a.graph)}(a.graph, id, a.dims)
        end
        Base.Broadcast.broadcasted(::typeof($op), a::_ADTensor) = _unary($op, a)
    end
end
Base.:-(a::_ADTensor) = _unary(-, a)
Base.conj(a::_ADTensor) = _unary(conj, a)

function _broadcast_to(a::EagerTensor{T}, dims) where {T}
    a.dims == dims && return a
    values = UInt[dims...]
    id = GC.@preserve values RustCall.result_to_exception(_rust_call(a.graph, RustCrate.broadcast_to, UInt(a.id), pointer(values), UInt(length(values))))
    EagerTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
end
function _broadcast_to(a::TracedTensor{T}, dims) where {T}
    a.dims == dims && return a
    values = UInt[dims...]
    id = GC.@preserve values RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_broadcast_to, UInt(a.id), pointer(values), UInt(length(values))))
    TracedTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
end
function _convert_tensor(a::EagerTensor{T,N}, ::Type{S}) where {T,N,S<:ADElement}
    T === S && return a
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.convert_node, UInt(a.id), _typed_kind(S)))
    EagerTensor{S,N,typeof(a.graph)}(a.graph, id, a.dims)
end
function _convert_tensor(a::TracedTensor{T,N}, ::Type{S}) where {T,N,S<:ADElement}
    T === S && return a
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_convert_node, UInt(a.id), _typed_kind(S)))
    TracedTensor{S,N,typeof(a.graph)}(a.graph, id, a.dims)
end

function _broadcast_pair(a::_ADTensor, b::_ADTensor)
    a.graph === b.graph || throw(ArgumentError("tensors belong to different contexts"))
    promoted = promote_type(eltype(a), eltype(b))
    if eltype(a) !== eltype(b)
        promoted <: ADElement || throw(ArgumentError("unsupported promoted tensor element type: $promoted"))
        a, b = _convert_tensor(a, promoted), _convert_tensor(b, promoted)
    end
    n = max(length(a.dims), length(b.dims))
    dims = ntuple(n) do i
        x = i <= length(a.dims) ? a.dims[i] : 1
        y = i <= length(b.dims) ? b.dims[i] : 1
        x == y ? x : x == 1 ? y : y == 1 ? x : throw(DimensionMismatch("broadcast dimensions do not match"))
    end
    return _broadcast_to(a, dims), _broadcast_to(b, dims)
end
for (op, eager_code, traced_code) in ((:+, 0, 0), (:-, 1, 1), (:*, 2, 3), (:/, 3, 4), (:^, 4, 5), (:rem, 5, 6), (:max, 6, 7), (:min, 7, 8))
    @eval begin
        function Base.Broadcast.broadcasted(::typeof($op), lhs::EagerTensor, rhs::EagerTensor)
            if $(op === :/)
                T = Base.promote_op(/, eltype(lhs), eltype(rhs))
                lhs, rhs = _convert_tensor(lhs, T), _convert_tensor(rhs, T)
            end
            a, b = _broadcast_pair(lhs, rhs)
            T = eltype(a)
            id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.binary, UInt(a.id), UInt(b.id), Int32($eager_code)))
            return EagerTensor{T,length(a.dims),typeof(a.graph)}(a.graph, id, a.dims)
        end
        function Base.Broadcast.broadcasted(::typeof($op), lhs::TracedTensor, rhs::TracedTensor)
            if $(op === :/)
                T = Base.promote_op(/, eltype(lhs), eltype(rhs))
                lhs, rhs = _convert_tensor(lhs, T), _convert_tensor(rhs, T)
            end
            a, b = _broadcast_pair(lhs, rhs)
            T = eltype(a)
            id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_binary, UInt(a.id), UInt(b.id), Int32($traced_code)))
            return TracedTensor{T,length(a.dims),typeof(a.graph)}(a.graph, id, a.dims)
        end
    end
end

function _scalar_tensor(a::EagerTensor{T}, b::Number) where {T}
    S = promote_type(T, typeof(b))
    S <: TensorElement || throw(ArgumentError("unsupported promoted tensor element type: $S"))
    EagerTensor(EagerContext(a.graph), fill(convert(S, b), a.dims))
end
function _scalar_tensor(a::TracedTensor{T}, b::Number) where {T}
    S = promote_type(T, typeof(b))
    S <: TensorElement || throw(ArgumentError("unsupported promoted tensor element type: $S"))
    TracedTensor(TracedContext(a.graph), fill(convert(S, b), a.dims))
end
for op in (:+, :-, :*, :/, :^, :rem, :max, :min)
    @eval begin
        Base.Broadcast.broadcasted(::typeof($op), a::_ADTensor, b::Number) = Base.Broadcast.broadcasted($op, a, _scalar_tensor(a, b))
        Base.Broadcast.broadcasted(::typeof($op), a::Number, b::_ADTensor) = Base.Broadcast.broadcasted($op, _scalar_tensor(b, a), b)
    end
end

function _shape_op(a::EagerTensor{T}, args, op, dims) where {T}
    values = UInt[args...]
    id = GC.@preserve values RustCall.result_to_exception(_rust_call(a.graph, RustCrate.shape_op, UInt(a.id), pointer(values), UInt(length(values)), Int32(op)))
    EagerTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
end
function _shape_op(a::TracedTensor{T}, args, op, dims) where {T}
    values = UInt[args...]
    id = GC.@preserve values RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_shape_op, UInt(a.id), pointer(values), UInt(length(values)), Int32(op)))
    TracedTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
end
function Base.reshape(a::_ADTensor, dims::Tuple{Vararg{Int}})
    all(>=(0), dims) && prod(dims) == prod(a.dims) || throw(DimensionMismatch("reshape dimensions must match tensor length"))
    _shape_op(a, dims, 0, dims)
end
Base.reshape(a::_ADTensor, dims::Int...) = reshape(a, dims)
function Base.permutedims(a::_ADTensor, perm)
    p = Tuple(perm)
    sort(collect(p)) == collect(1:length(a.dims)) || throw(ArgumentError("invalid permutation"))
    _shape_op(a, p .- 1, 1, ntuple(i -> a.dims[p[i]], length(p)))
end
function Base.permutedims(a::_ADTensor)
    length(a.dims) == 1 && return reshape(a, (1, a.dims[1]))
    length(a.dims) == 2 && return permutedims(a, (2, 1))
    throw(ArgumentError("permutedims without a permutation requires a vector or matrix"))
end
function Base.transpose(a::_ADTensor)
    length(a.dims) == 1 && return reshape(a, (1, a.dims[1]))
    length(a.dims) == 2 && return permutedims(a, (2, 1))
    throw(ArgumentError("transpose requires a vector or matrix"))
end
Base.adjoint(a::_ADTensor) = conj(transpose(a))

Base.Broadcast.broadcasted(::typeof(Base.literal_pow), ::typeof(^), a::_ADTensor, ::Val{P}) where {P} = Base.Broadcast.broadcasted(^, a, P)
function _reduce_dims(a::_ADTensor, dims, op)
    axes = dims isa Integer ? (dims,) : Tuple(dims)
    all(d -> d isa Integer && d >= 1, axes) || throw(ArgumentError("dimensions must be positive integers"))
    length(unique(axes)) == length(axes) || throw(ArgumentError("dimensions must be unique"))
    axes = Tuple(d for d in axes if d <= length(a.dims))
    isempty(axes) && return a
    reduced = Tuple(a.dims[i] for i in eachindex(a.dims) if !(i in axes))
    retained = ntuple(i -> i in axes ? 1 : a.dims[i], length(a.dims))
    reshape(_shape_op(a, axes .- 1, op, reduced), retained)
end

function _reduce_all(a::EagerTensor{T}, op) where {T}
    y = _shape_op(a, ntuple(i -> i - 1, length(a.dims)), op, ())
    EagerScalar{T,typeof(a.graph)}(a.graph, y.id)
end
function _reduce_all(a::TracedTensor, op)
    _shape_op(a, ntuple(i -> i - 1, length(a.dims)), op, ())
end
for (op, code) in ((:prod, 4), (:maximum, 5), (:minimum, 6))
    @eval function Base.$op(a::_ADTensor; dims=:)
        dims isa Colon ? _reduce_all(a, $code) : _reduce_dims(a, dims, $code)
    end
end
function Base.reverse(a::_ADTensor; dims=:)
    axes = dims isa Colon ? Tuple(1:length(a.dims)) : dims isa Integer ? (dims,) : Tuple(dims)
    all(d -> d isa Integer && 1 <= d <= length(a.dims), axes) || throw(ArgumentError("invalid dimensions"))
    length(unique(axes)) == length(axes) || throw(ArgumentError("dimensions must be unique"))
    _shape_op(a, axes .- 1, 3, a.dims)
end

function Statistics.mean(a::_ADTensor; dims=:)
    dims isa Colon ? _reduce_all(a, 7) : _reduce_dims(a, dims, 7)
end

for op in (:*, :/)
    @eval Base.$op(a::_ADTensor, b::Number) = Base.Broadcast.broadcasted($op, a, b)
end
Base.:*(a::Number, b::_ADTensor) = Base.Broadcast.broadcasted(*, a, b)
function Base.Broadcast.broadcasted(::typeof(abs), a::EagerTensor{T,N}) where {T<:TensorElement,N}
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.unary, UInt(a.id), Int32(11)))
    EagerTensor{typeof(real(zero(T))),N,typeof(a.graph)}(a.graph, id, a.dims)
end
function Base.Broadcast.broadcasted(::typeof(abs), a::TracedTensor{T,N}) where {T<:TensorElement,N}
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_unary, UInt(a.id), Int32(11)))
    TracedTensor{typeof(real(zero(T))),N,typeof(a.graph)}(a.graph, id, a.dims)
end

function _slice(a::EagerTensor{T}, values, dims) where {T}
    id = GC.@preserve values RustCall.result_to_exception(_rust_call(a.graph, RustCrate.slice_node, UInt(a.id), pointer(values), UInt(length(a.dims))))
    EagerTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
end
function _slice(a::TracedTensor{T}, values, dims) where {T}
    id = GC.@preserve values RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_slice_node, UInt(a.id), pointer(values), UInt(length(a.dims))))
    TracedTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
end
function _slice_index(a::_ADTensor, indices::Union{Integer,AbstractRange{<:Integer},Colon}...)
    length(indices) == length(a.dims) || throw(ArgumentError("one index per dimension is required"))
    ranges = map(indices, a.dims) do i, n
        r = i isa Colon ? (1:n) : i isa Integer ? (i:i) : i
        isempty(r) && throw(ArgumentError("empty slices are not supported"))
        step(r) > 0 || throw(ArgumentError("slice steps must be positive"))
        first(r) >= 1 && last(r) <= n || throw(BoundsError(a, indices))
        r
    end
    starts = map(r -> first(r)-1, ranges)
    limits = map(last, ranges)
    strides = map(step, ranges)
    values = UInt[starts..., limits..., strides...]
    full_dims = map(length, ranges)
    result = _slice(a, values, full_dims)
    out_dims = Tuple(full_dims[i] for i in eachindex(indices) if !(indices[i] isa Integer))
    output = full_dims == out_dims ? result : reshape(result, out_dims)
    isempty(out_dims) && output isa EagerTensor && return _scalar_result(output)
    return output
end

function _matrix_structure(a::EagerTensor{T,2}, op, k, dims) where {T}
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.matrix_structure, UInt(a.id), Int32(op), Int64(k)))
    EagerTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
end
function _matrix_structure(a::TracedTensor{T,2}, op, k, dims) where {T}
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_matrix_structure, UInt(a.id), Int32(op), Int64(k)))
    TracedTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
end
LinearAlgebra.tril(a::Union{EagerTensor{T,2},TracedTensor{T,2}}, k::Integer=0) where {T} = _matrix_structure(a, 0, k, a.dims)
LinearAlgebra.triu(a::Union{EagerTensor{T,2},TracedTensor{T,2}}, k::Integer=0) where {T} = _matrix_structure(a, 1, k, a.dims)
LinearAlgebra.diag(a::Union{EagerTensor{T,2},TracedTensor{T,2}}) where {T} = _matrix_structure(a, 2, 0, (min(a.dims...),))

function _linalg_unary(a::EagerTensor{T,2}, op, dims) where {T}
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.linalg_unary, UInt(a.id), Int32(op)))
    isempty(dims) ? EagerScalar{T,typeof(a.graph)}(a.graph, id) : EagerTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
end
function _linalg_unary(a::TracedTensor{T,2}, op, dims) where {T}
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_linalg_unary, UInt(a.id), Int32(op)))
    TracedTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
end
LinearAlgebra.det(a::Union{EagerTensor{T,2},TracedTensor{T,2}}) where {T} = _linalg_unary(a, 0, ())
Base.inv(a::Union{EagerTensor{T,2},TracedTensor{T,2}}) where {T} = _linalg_unary(a, 1, a.dims)
LinearAlgebra.pinv(a::Union{EagerTensor{T,2},TracedTensor{T,2}}) where {T} = _linalg_unary(a, 2, reverse(a.dims))
function Base.:\(a::EagerTensor{T,2}, b::EagerTensor{T,2}) where {T}
    a.graph === b.graph || throw(ArgumentError("tensors belong to different contexts"))
    a.dims[1] == b.dims[1] || throw(DimensionMismatch("solve dimensions do not match"))
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.linalg_solve, UInt(a.id), UInt(b.id), a.dims[1] != a.dims[2]))
    EagerTensor{T,2,typeof(a.graph)}(a.graph, id, (a.dims[2], b.dims[2]))
end
function Base.:\(a::TracedTensor{T,2}, b::TracedTensor{T,2}) where {T}
    a.graph === b.graph || throw(ArgumentError("tensors belong to different contexts"))
    a.dims[1] == b.dims[1] || throw(DimensionMismatch("solve dimensions do not match"))
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_linalg_solve, UInt(a.id), UInt(b.id), a.dims[1] != a.dims[2]))
    TracedTensor{T,2,typeof(a.graph)}(a.graph, id, (a.dims[2], b.dims[2]))
end

function _factor(a::EagerTensor{T,2}, op) where {T}
    RustCall.result_to_exception(_rust_call(a.graph, RustCrate.linalg_factor, UInt(a.id), Int32(op)))
end
function _factor(a::TracedTensor{T,2}, op) where {T}
    RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_linalg_factor, UInt(a.id), Int32(op)))
end
_factor_tensor(a::EagerTensor, ::Type{T}, id, dims) where {T} = EagerTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
_factor_tensor(a::TracedTensor, ::Type{T}, id, dims) where {T} = TracedTensor{T,length(dims),typeof(a.graph)}(a.graph, id, dims)
function LinearAlgebra.svd(a::Union{EagerTensor{T,2},TracedTensor{T,2}}; full::Bool=false) where {T<:ADElement}
    full && throw(ArgumentError("full SVD is not supported for differentiation"))
    m, n = a.dims; k = min(m, n)
    id = _factor(a, 0)
    TensorSVD(_factor_tensor(a, T, id, (m,k)), _factor_tensor(a, typeof(real(zero(T))), id+1, (k,)), _factor_tensor(a, T, id+2, (k,n)))
end
function LinearAlgebra.qr(a::Union{EagerTensor{T,2},TracedTensor{T,2}}) where {T<:ADElement}
    m, n = a.dims; k = min(m, n)
    id = _factor(a, 1)
    TensorQR(_factor_tensor(a, T, id, (m,k)), _factor_tensor(a, T, id+1, (k,n)))
end

function _cat(first::_ADTensor, rest, dims)
    dims isa Integer && 1 <= dims <= length(first.dims) || throw(ArgumentError("cat requires one existing dimension"))
    for a in rest
        a.graph === first.graph || throw(ArgumentError("tensors belong to different contexts"))
        length(a.dims) == length(first.dims) || throw(DimensionMismatch("tensor ranks do not match"))
        all(i -> i == dims || a.dims[i] == first.dims[i], eachindex(first.dims)) || throw(DimensionMismatch("cat dimensions do not match"))
    end
    inputs = (first, rest...)
    promoted = promote_type(map(eltype, inputs)...)
    if any(a -> eltype(a) !== promoted, inputs)
        promoted <: ADElement || throw(ArgumentError("unsupported promoted tensor element type: $promoted"))
        inputs = map(a -> _convert_tensor(a, promoted), inputs)
    end
    shape = ntuple(i -> i == dims ? sum(a -> a.dims[i], inputs) : first.dims[i], length(first.dims))
    ids = UInt[a.id for a in inputs]
    fn = first isa EagerTensor ? RustCrate.concatenate : RustCrate.traced_concatenate
    id = GC.@preserve ids RustCall.result_to_exception(_rust_call(first.graph, fn, pointer(ids), UInt(length(ids)), UInt(dims - 1)))
    _factor_tensor(first, promoted, id, shape)
end
Base.cat(first::_ADTensor, rest::_ADTensor...; dims) = _cat(first, rest, dims)

Base.Broadcast.broadcasted(::typeof(clamp), a::_ADTensor, lo::Union{_ADTensor,Number,AbstractArray}, hi::Union{_ADTensor,Number,AbstractArray}) = Base.Broadcast.broadcasted(min, Base.Broadcast.broadcasted(max, a, lo), hi)

function LinearAlgebra.cholesky(a::Union{EagerTensor{T,2},TracedTensor{T,2}}) where {T<:ADElement}
    a.dims[1] == a.dims[2] || throw(DimensionMismatch("cholesky requires a square matrix"))
    ishermitian(Array(a)) || throw(PosDefException(-1))
    TensorCholesky(_linalg_unary(a, 3, a.dims))
end

function LinearAlgebra.eigen(a::Union{EagerTensor{T,2},TracedTensor{T,2}}; sortby=LinearAlgebra.eigsortby) where {T<:ADElement}
    a.dims[1] == a.dims[2] || throw(DimensionMismatch("eigen requires a square matrix"))
    n = a.dims[1]
    id = _factor(a, 2)
    C = Complex{typeof(real(zero(T)))}
    _sort_eigen(TensorEigen(_factor_tensor(a, C, id, (n,)), _factor_tensor(a, C, id+1, (n,n))), sortby)
end

function LinearAlgebra.norm(a::EagerTensor{T}) where {T<:ADElement}
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.norm_node, UInt(a.id)))
    EagerScalar{typeof(real(zero(T))),typeof(a.graph)}(a.graph, id)
end
function LinearAlgebra.norm(a::TracedTensor{T}) where {T<:ADElement}
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_norm_node, UInt(a.id)))
    TracedTensor{typeof(real(zero(T))),0,typeof(a.graph)}(a.graph, id, ())
end

function LinearAlgebra.Hermitian(a::Union{EagerTensor{T,2},TracedTensor{T,2}}, uplo::Symbol=:U) where {T<:ADElement}
    uplo in (:U, :L) || throw(ArgumentError("uplo must be :U or :L"))
    a.dims[1] == a.dims[2] || throw(DimensionMismatch("Hermitian requires a square matrix"))
    TensorHermitian(a, uplo)
end
function _hermitian_matrix(h::TensorHermitian)
    diagonal = tril(triu(h.data))
    diagonal = (diagonal + conj(diagonal)) / 2
    triangle = h.uplo === :U ? triu(h.data, 1) : tril(h.data, -1)
    diagonal + triangle + adjoint(triangle)
end
function LinearAlgebra.eigen(h::TensorHermitian; sortby=LinearAlgebra.eigsortby)
    a = _hermitian_matrix(h)
    n = a.dims[1]; T = eltype(a)
    id = _factor(a, 3)
    _sort_eigen(TensorEigen(_factor_tensor(a, typeof(real(zero(T))), id, (n,)), _factor_tensor(a, T, id+1, (n,n))), sortby)
end
LinearAlgebra.cholesky(h::TensorHermitian) = TensorCholesky(_linalg_unary(_hermitian_matrix(h), 3, h.data.dims))

Base.getindex(a::_ADTensor, indices::Union{Integer,AbstractRange{<:Integer},Colon}...) = _slice_index(a, indices...)
Base.getindex(a::EagerTensor{T,1}, indices::AbstractRange{<:Integer}) where {T} = _slice_index(a, indices)
Base.getindex(a::TracedTensor{T,1}, indices::AbstractRange{<:Integer}) where {T} = _slice_index(a, indices)

function Base.getindex(a::EagerTensor{T,1}, indices::AbstractVector{<:Integer}) where {T}
    all(i -> 1 <= i <= a.dims[1], indices) || throw(BoundsError(a, indices))
    ix = EagerTensor(EagerContext(a.graph), reshape(Int64.(indices) .- 1, :, 1))
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.gather_vector, UInt(a.id), UInt(ix.id)))
    EagerTensor{T,1,typeof(a.graph)}(a.graph, id, (length(indices),))
end
function Base.getindex(a::TracedTensor{T,1}, indices::AbstractVector{<:Integer}) where {T}
    all(i -> 1 <= i <= a.dims[1], indices) || throw(BoundsError(a, indices))
    ix = TracedTensor(TracedContext(a.graph), reshape(Int64.(indices) .- 1, :, 1))
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_gather_vector, UInt(a.id), UInt(ix.id)))
    TracedTensor{T,1,typeof(a.graph)}(a.graph, id, (length(indices),))
end

function LinearAlgebra.lu(a::Union{EagerTensor{T,2},TracedTensor{T,2}}) where {T<:ADElement}
    m, n = a.dims; k = min(m, n)
    id = _factor(a, 4)
    TensorLU(_factor_tensor(a, T, id, (m,m)), _factor_tensor(a, T, id+1, (m,k)), _factor_tensor(a, T, id+2, (k,n)))
end

Base.:\(a::EagerTensor{T,2}, b::EagerTensor{T,1}) where {T} = reshape(a \ reshape(b, (b.dims[1], 1)), (a.dims[2],))
Base.:\(a::TracedTensor{T,2}, b::TracedTensor{T,1}) where {T} = reshape(a \ reshape(b, (b.dims[1], 1)), (a.dims[2],))
Base.:\(a::EagerTensor{T,2}, b::AbstractArray{T,N}) where {T,N} = a \ EagerTensor(EagerContext(a.graph), b)
Base.:\(a::TracedTensor{T,2}, b::AbstractArray{T,N}) where {T,N} = a \ TracedTensor(TracedContext(a.graph), b)

_constant_tensor(a::EagerTensor, b::AbstractArray{T}) where {T<:TensorElement} = EagerTensor(EagerContext(a.graph), b)
_constant_tensor(a::TracedTensor, b::AbstractArray{T}) where {T<:TensorElement} = TracedTensor(TracedContext(a.graph), b)
for op in (:+, :-, :*, :/, :^, :rem, :max, :min)
    @eval begin
        Base.Broadcast.broadcasted(::typeof($op), a::_ADTensor, b::AbstractArray) = Base.Broadcast.broadcasted($op, a, _constant_tensor(a, b))
        Base.Broadcast.broadcasted(::typeof($op), a::AbstractArray, b::_ADTensor) = Base.Broadcast.broadcasted($op, _constant_tensor(b, a), b)
    end
end

Base.:*(a::Union{EagerTensor{T,2},TracedTensor{T,2}}, b::AbstractMatrix{S}) where {T,S<:TensorElement} = a * _constant_tensor(a, b)
Base.:*(a::AbstractMatrix{S}, b::Union{EagerTensor{T,2},TracedTensor{T,2}}) where {T,S<:TensorElement} = _constant_tensor(b, a) * b

function Base.getproperty(f::TensorSVD, name::Symbol)
    name === :V && return adjoint(getfield(f, :Vt))
    getfield(f, name)
end
Base.propertynames(::TensorSVD, private::Bool=false) = (:U, :S, :V, :Vt)
Base.iterate(f::TensorSVD, state=1) = state > 3 ? nothing : ((f.U, f.S, f.V)[state], state + 1)
Base.iterate(f::TensorQR, state=1) = state > 2 ? nothing : ((f.Q, f.R)[state], state + 1)
Base.iterate(f::TensorEigen, state=1) = state > 2 ? nothing : ((f.values, f.vectors)[state], state + 1)
function Base.getproperty(f::TensorCholesky, name::Symbol)
    name === :U && return adjoint(getfield(f, :L))
    getfield(f, name)
end
Base.propertynames(::TensorCholesky, private::Bool=false) = (:L, :U)

_scalar_tensor_value(a::EagerScalar{T}) where {T} = EagerTensor{T,0,typeof(a.graph)}(a.graph, a.id, ())
_scalar_result(a::EagerTensor{T,0}) where {T} = EagerScalar{T,typeof(a.graph)}(a.graph, a.id)
for op in (:+, :-, :*, :/, :^, :max, :min)
    @eval begin
        Base.$op(a::EagerScalar, b::EagerScalar) = _scalar_result(Base.Broadcast.broadcasted($op, _scalar_tensor_value(a), _scalar_tensor_value(b)))
        Base.$op(a::EagerScalar, b::Number) = _scalar_result(Base.Broadcast.broadcasted($op, _scalar_tensor_value(a), b))
        Base.$op(a::Number, b::EagerScalar) = _scalar_result(Base.Broadcast.broadcasted($op, a, _scalar_tensor_value(b)))
        Base.$op(a::TracedTensor{T,0}, b::Number) where {T} = Base.Broadcast.broadcasted($op, a, b)
        Base.$op(a::Number, b::TracedTensor{T,0}) where {T} = Base.Broadcast.broadcasted($op, a, b)
    end
end
Base.:-(a::EagerScalar) = _scalar_result(_unary(-, _scalar_tensor_value(a)))
for op in (:exp, :log, :sqrt, :sin, :cos, :tanh, :expm1, :log1p, :conj, :sign)
    @eval begin
        Base.$op(a::EagerScalar) = _scalar_result(_unary($op, _scalar_tensor_value(a)))
        Base.$op(a::TracedTensor{T,0}) where {T} = _unary($op, a)
    end
end

function Base.reshape(a::_ADTensor, dims::Tuple{Vararg{Union{Int,Colon}}})
    inferred = size(reshape(Base.OneTo(prod(a.dims)), dims))
    reshape(a, inferred)
end
Base.reshape(a::_ADTensor, dims::Union{Int,Colon}...) = reshape(a, dims)

function LinearAlgebra.diag(a::Union{EagerTensor{T,2},TracedTensor{T,2}}, k::Integer) where {T}
    k == 0 && return diag(a)
    m, n = a.dims
    if k > 0
        k < n || throw(ArgumentError("empty diagonals are not supported"))
        return diag(a[:, (k + 1):n])
    end
    -k < m || throw(ArgumentError("empty diagonals are not supported"))
    return diag(a[(1 - k):m, :])
end

function LinearAlgebra.LowerTriangular(a::Union{EagerTensor{T,2},TracedTensor{T,2}}) where {T}
    a.dims[1] == a.dims[2] || throw(DimensionMismatch("triangular matrix must be square"))
    TensorTriangular(a, true)
end
function LinearAlgebra.UpperTriangular(a::Union{EagerTensor{T,2},TracedTensor{T,2}}) where {T}
    a.dims[1] == a.dims[2] || throw(DimensionMismatch("triangular matrix must be square"))
    TensorTriangular(a, false)
end
Base.:\(a::TensorTriangular, b::Union{_ADTensor,AbstractArray}) = (a.lower ? tril(a.data) : triu(a.data)) \ b

for op in (:+, :-)
    @eval begin
        function Base.$op(a::EagerTensor{T,N}, b::EagerTensor{S,M}) where {T,S,N,M}
            a.dims == b.dims || throw(DimensionMismatch("tensor shapes do not match"))
            Base.Broadcast.broadcasted($op, a, b)
        end
        function Base.$op(a::TracedTensor{T,N}, b::TracedTensor{S,M}) where {T,S,N,M}
            a.dims == b.dims || throw(DimensionMismatch("tensor shapes do not match"))
            Base.Broadcast.broadcasted($op, a, b)
        end
    end
end

function Base.:*(a::EagerTensor{T,2}, b::EagerTensor{S,2}) where {T,S}
    promoted = promote_type(T, S)
    promoted <: ADElement || throw(ArgumentError("unsupported promoted tensor element type: $promoted"))
    _convert_tensor(a, promoted) * _convert_tensor(b, promoted)
end
function Base.:*(a::TracedTensor{T,2}, b::TracedTensor{S,2}) where {T,S}
    promoted = promote_type(T, S)
    promoted <: ADElement || throw(ArgumentError("unsupported promoted tensor element type: $promoted"))
    _convert_tensor(a, promoted) * _convert_tensor(b, promoted)
end

function Base.:*(a::EagerTensor{T,2}, x::EagerTensor{S,1}) where {T,S}
    a.dims[2] == x.dims[1] || throw(DimensionMismatch("matrix and vector dimensions do not match"))
    reshape(a * reshape(x, (x.dims[1], 1)), (a.dims[1],))
end
function Base.:*(a::TracedTensor{T,2}, x::TracedTensor{S,1}) where {T,S}
    a.dims[2] == x.dims[1] || throw(DimensionMismatch("matrix and vector dimensions do not match"))
    reshape(a * reshape(x, (x.dims[1], 1)), (a.dims[1],))
end
Base.:*(a::AbstractMatrix{T}, x::EagerTensor{S,1}) where {T<:TensorElement,S} = _constant_tensor(x, a) * x
Base.:*(a::AbstractMatrix{T}, x::TracedTensor{S,1}) where {T<:TensorElement,S} = _constant_tensor(x, a) * x

for op in (:+, :-)
    @eval begin
        function Base.$op(a::_ADTensor, b::AbstractArray{T}) where {T<:TensorElement}
            a.dims == size(b) || throw(DimensionMismatch("tensor shapes do not match"))
            $op(a, _constant_tensor(a, b))
        end
        function Base.$op(a::AbstractArray{T}, b::_ADTensor) where {T<:TensorElement}
            size(a) == b.dims || throw(DimensionMismatch("tensor shapes do not match"))
            $op(_constant_tensor(b, a), b)
        end
    end
end

function Base.getindex(a::EagerTensor{T,1}, mask::AbstractVector{Bool}) where {T}
    length(mask) == a.dims[1] || throw(BoundsError(a, mask))
    a[findall(mask)]
end
function Base.getindex(a::TracedTensor{T,1}, mask::AbstractVector{Bool}) where {T}
    length(mask) == a.dims[1] || throw(BoundsError(a, mask))
    a[findall(mask)]
end

Base.sum(a::EagerScalar) = a

Base.size(a::_ADTensor) = a.dims
function Base.size(a::_ADTensor, d::Integer)
    d >= 1 || throw(ArgumentError("dimension must be positive"))
    d <= length(a.dims) ? a.dims[d] : 1
end
Base.ndims(::Union{EagerTensor{T,N},TracedTensor{T,N}}) where {T,N} = N
Base.length(a::_ADTensor) = prod(a.dims)
Base.axes(a::_ADTensor) = map(Base.OneTo, a.dims)
Base.axes(a::_ADTensor, d::Integer) = Base.OneTo(size(a, d))

function _sort_eigen(f::TensorEigen, sortby)
    sortby === nothing && return f
    permutation = sortperm(Array(f.values); by=sortby)
    all(i -> permutation[i] == i, eachindex(permutation)) && return f
    values = f.values[permutation]
    columns = map(i -> f.vectors[:, i:i], permutation)
    vectors = length(columns) == 1 ? only(columns) : cat(columns...; dims=2)
    TensorEigen(values, vectors)
end

Base.getindex(a::EagerTensor{T,1}, i::Integer) where {T} = _scalar_result(reshape(a[[i]], ()))
Base.getindex(a::TracedTensor{T,1}, i::Integer) where {T} = reshape(a[[i]], ())

# The core Rust `sign` rule treats sign as locally constant. For complex values,
# sign(z) = z / abs(z) has a nonzero real-linear derivative away from zero.
# Adding 1 - sign(abs(z)) makes the denominator one only at z == 0.
function _complex_sign(a::Union{EagerTensor{T},TracedTensor{T}}) where {T<:Complex}
    magnitude = Base.Broadcast.broadcasted(abs, a)
    iszero = Base.Broadcast.broadcasted(-, one(eltype(magnitude)), Base.Broadcast.broadcasted(sign, magnitude))
    denominator = Base.Broadcast.broadcasted(+, magnitude, iszero)
    Base.Broadcast.broadcasted(/, a, denominator)
end
_unary(::typeof(sign), a::EagerTensor{T}) where {T<:Complex} = _complex_sign(a)
_unary(::typeof(sign), a::TracedTensor{T}) where {T<:Complex} = _complex_sign(a)
