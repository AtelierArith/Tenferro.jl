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

"""
    Tenferro.num_threads() -> Int

Number of CPU worker threads tenferro-rs kernels run on.

All tenferro-rs operations of the process share one CPU execution engine, which keeps its
buffer pool and kernel plan cache between calls. Its thread count is taken from the
`TENFERRO_NUM_THREADS` environment variable when the engine is first used (default `1`,
serial) and can be changed with [`Tenferro.set_num_threads`](@ref).
"""
num_threads() = Int(RustCrate.num_threads())

"""
    Tenferro.set_num_threads(n::Integer) -> Int

Run tenferro-rs kernels on `n ≥ 1` CPU worker threads (`1` is serial) and return the new
thread count. The shared engine is rebuilt with a new worker pool, so its buffer pool and
kernel plan cache start empty. Tensors and contexts created before the change remain valid.

Threaded kernels may sum in a different order than serial ones, so results can differ in
the last bits. Julia threads that use tenferro concurrently take turns on the engine.
"""
function set_num_threads(n::Integer)
    n >= 1 || throw(ArgumentError("number of threads must be at least 1, got $n"))
    return Int(RustCrate.set_num_threads(UInt(n)))
end
