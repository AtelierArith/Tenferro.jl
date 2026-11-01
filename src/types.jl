"""
    AutoTenferro()

DifferentiationInterface backend for first-order gradients of scalar-valued
functions built from supported tenferro operations on CPU arrays of Float32, Float64, ComplexF32, or ComplexF64.
Supports addition, subtraction, matrix multiplication, and `sum`; gradients of
scalar outputs and array-output pullbacks for tuples of array inputs.
"""
struct AutoTenferro <: AbstractADType end

"""
    EagerTensor(ctx::EagerContext, data::AbstractArray{T})

An eager tenferro-rs tensor. Construction copies `data` into the given context.
Tensors used in the same operation must belong to the same context.
"""
struct EagerTensor{T,N,G}
    graph::G
    id::Int64
    dims::NTuple{N,Int}
end

"""Owns an eager execution context shared by its tensors."""
struct EagerContext{G}
    graph::G
end
EagerContext() = EagerContext(_RustState(RustCrate.EagerGraph()))
Base.close(ctx::EagerContext) = _close_state(ctx.graph)

const ADElement = Union{Float32,Float64,ComplexF32,ComplexF64}
const TensorElement = Union{ADElement,Int32,Int64,Bool}

function EagerTensor(ctx::EagerContext, x::AbstractArray{T,N}) where {T<:TensorElement,N}
    id = _with_input_buffer(x) do data, shape, rank
        RustCall.result_to_exception(_rust_call(ctx.graph, RustCrate.input_array, data, shape, rank, _typed_kind(T)))
    end
    id >= 0 || error("tenferro-rs could not create the input tensor")
    return EagerTensor{T,N,typeof(ctx.graph)}(ctx.graph, id, size(x))
end

struct EagerScalar{T,G}
    graph::G
    id::Int64
end

"""Owns the graph containing a collection of traced tensors."""
struct TracedContext{G}
    graph::G
end
TracedContext() = TracedContext(_RustState(RustCrate.TracedGraph()))
Base.close(ctx::TracedContext) = _close_state(ctx.graph)

"""
    TracedTensor(ctx::TracedContext, data::AbstractArray{T})

A node in a tenferro-rs computation graph for tenferro-rs built-in CPU element types.
Construction copies the input; `Array(tensor)` compiles and executes the graph.
"""
struct TracedTensor{T,N,G}
    graph::G
    id::Int64
    dims::NTuple{N,Int}
end

function TracedTensor(ctx::TracedContext, x::AbstractArray{T,N}) where {T<:TensorElement,N}
    id = _with_input_buffer(x) do data, shape, rank
        RustCall.result_to_exception(_rust_call(ctx.graph, RustCrate.traced_input_array, data, shape, rank, _typed_kind(T)))
    end
    id >= 0 || error("tenferro-rs could not create the traced input")
    return TracedTensor{T,N,typeof(ctx.graph)}(ctx.graph, id, size(x))
end

"""DifferentiationInterface backend using tenferro-rs traced graph transformations."""
struct AutoTenferroTraced <: AbstractADType end

"""
    TypedTensor(array::AbstractArray)

A CPU tensor backed by tenferro-rs and implementing Julia's `AbstractArray`
interface. Construction copies values in column-major order; `Array(tensor)`
returns another independent copy. Supports arbitrary dimensionality and element
types `Float32`, `Float64`, `Int32`, `Int64`, `Bool`, `ComplexF32`, and `ComplexF64`.
"""
struct TypedTensor{T,N,S} <: AbstractArray{T,N}
    storage::S
    dims::NTuple{N,Int}
end

const TypedElement = Union{Float32,Float64,Int32,Int64,Bool,ComplexF32,ComplexF64}
_typed_kind(::Type{Float32}) = Int32(0)
_typed_kind(::Type{Float64}) = Int32(1)
_typed_kind(::Type{Int32}) = Int32(2)
_typed_kind(::Type{Int64}) = Int32(3)
_typed_kind(::Type{ComplexF32}) = Int32(4)
_typed_kind(::Type{ComplexF64}) = Int32(5)
_typed_kind(::Type{Bool}) = Int32(6)

function TypedTensor(x::AbstractArray{T,N}) where {T<:TypedElement,N}
    dims = size(x)
    storage = _with_input_buffer(x) do data, shape, rank
        _RustState(RustCrate.TypedStorage(data, shape, rank, _typed_kind(T)))
    end
    _rust_call(storage, RustCrate.typed_valid) || error("tenferro-rs could not create typed tensor: ", _rust_call(storage, RustCrate.typed_error))
    return TypedTensor{T,N,typeof(storage)}(storage, dims)
end
TypedTensor(x::AbstractArray) = throw(ArgumentError("unsupported TypedTensor element type: $(eltype(x))"))



Base.eltype(::Type{<:EagerTensor{T}}) where {T} = T
Base.eltype(::Type{<:EagerScalar{T}}) where {T} = T
Base.eltype(::Type{<:TracedTensor{T}}) where {T} = T
