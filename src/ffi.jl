# Raw pointers passed to Rust are valid only for the duration of each callback.
# Array(x) materializes a dense, column-major owner for views and other arrays.
function _with_input_buffer(f, x::AbstractArray)
    data = Array(x)
    shape = UInt[size(data)...]
    GC.@preserve data shape begin
        return f(Ptr{UInt8}(pointer(data)), pointer(shape), UInt(ndims(data)))
    end
end

function _with_output_buffer(f, out::Array)
    GC.@preserve out begin
        return f(Ptr{UInt8}(pointer(out)), UInt(length(out)))
    end
end

function _with_scalar_buffer(f, value::Ref{T}) where {T}
    GC.@preserve value begin
        return f(Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)))
    end
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
