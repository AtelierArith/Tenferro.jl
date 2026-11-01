function Base.Array(x::EagerTensor{T}) where {T}
    out = Array{T}(undef, x.dims)
    _with_output_buffer(out) do data, len
        RustCall.result_to_exception(_rust_call(x.graph, RustCrate.copy_array, UInt(x.id), data, len, _typed_kind(T)))
    end
    return out
end

for (op, code) in ((:+, 0), (:-, 1))
    @eval function Base.$op(a::EagerTensor{T,N}, b::EagerTensor{T,N}) where {T,N}
        return _apply_rule($op, (a, b), () -> begin
            a.graph === b.graph || throw(ArgumentError("tensors belong to different runtimes"))
            a.dims == b.dims || throw(DimensionMismatch("tensor shapes do not match"))
            id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.binary, UInt(a.id), UInt(b.id), Int32($code)))
            id >= 0 || error("tenferro-rs operation failed")
            EagerTensor{T,N,typeof(a.graph)}(a.graph, id, a.dims)
        end)
    end
end

function Base.:*(a::EagerTensor{T,2}, b::EagerTensor{T,2}) where {T}
    return _apply_rule(*, (a, b), () -> begin
        a.graph === b.graph || throw(ArgumentError("tensors belong to different runtimes"))
        a.dims[2] == b.dims[1] || throw(DimensionMismatch("matrix dimensions do not match"))
        id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.matmul, UInt(a.id), UInt(b.id)))
        id >= 0 || error("tenferro-rs matrix multiplication failed")
        EagerTensor{T,2,typeof(a.graph)}(a.graph, id, (a.dims[1], b.dims[2]))
    end)
end

function Base.sum(a::EagerTensor{T}) where {T}
    return _apply_rule(sum, (a,), () -> begin
        id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.sum, UInt(a.id)))
        id >= 0 || error("tenferro-rs reduction failed")
        EagerScalar{T,typeof(a.graph)}(a.graph, id)
    end)
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

function Base.sum(a::TracedTensor{T}) where {T}
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
