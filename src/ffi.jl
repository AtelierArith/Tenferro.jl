# RustCall borrows these vectors as checked `&[T]` / `&mut [T]` slices for the
# duration of one synchronous call; Rust copies anything it keeps.
_dense_vector(x::Array) = vec(x)
_dense_vector(x::AbstractArray) = vec(Array(x))

# Tensor payloads cross the boundary as native-endian bytes tagged by `_typed_kind`.
_bytes(x::AbstractArray) = reinterpret(UInt8, _dense_vector(x))
_dims(x::AbstractArray) = UInt[size(x)...]

# Mutable slices must be initialized before they are borrowed.
function _read_array(f, ::Type{T}, dims) where {T}
    out = zeros(T, dims)
    f(_bytes(out))
    return out
end

# RustCall exposes mutable Rust objects; serialize all accesses to each object.
mutable struct _RustState{T}
    value::Union{Nothing,T}
    lock::ReentrantLock
end
_RustState(value) = _RustState(value, ReentrantLock())
function _rust_call(state::_RustState, f, args...)
    lock(state.lock) do
        value = state.value
        isnothing(value) && throw(ArgumentError("tenferro context has been closed"))
        f(value, args...)
    end
end

function _close_state(state::_RustState)
    lock(state.lock) do
        state.value = nothing
    end
    return nothing
end
