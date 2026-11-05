# Raw pointers passed to Rust are valid only for the duration of each callback.
# RustCall supplies a contiguous copy, including for views and empty inputs.
function _with_input_buffer(f, x::AbstractArray)
    shape = UInt[size(x)...]
    RustCall.with_numeric_buffer(x) do data, _
        RustCall.with_numeric_buffer(shape) do dims, rank
            f(Ptr{UInt8}(data), dims, rank)
        end
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
