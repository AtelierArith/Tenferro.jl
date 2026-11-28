const DI = DifferentiationInterface

const _Backend = Union{AutoTenferro,AutoTenferroTraced}
const ADArray = AbstractArray{<:DifferentiableElement}
# Differentiable inputs: supported arrays, or Tuple/NamedTuple trees of them.
const _DiffInput = Union{ADArray,Tuple,NamedTuple}
const _Constants = Vararg{DI.Constant}

ADTypes.mode(::_Backend) = ADTypes.ReverseMode()
DI.inplace_support(::_Backend) = DI.InPlaceNotSupported()

# Input trees -------------------------------------------------------------------

_new_context(::AutoTenferro) = EagerContext()
_new_context(::AutoTenferroTraced) = TracedContext()
_input_tensor(ctx::EagerContext, x::ADArray) = EagerTensor(ctx, x)
_input_tensor(ctx::TracedContext, x::ADArray) = TracedTensor(ctx, x)

# Supported array leaves become tensors; any other leaf is passed through.
_tree_inputs(ctx, x::ADArray) = _input_tensor(ctx, x)
_tree_inputs(ctx, x::Tuple) = map(v -> _tree_inputs(ctx, v), x)
_tree_inputs(ctx, x::NamedTuple{K}) where {K} = NamedTuple{K}(_tree_inputs(ctx, Tuple(x)))
_tree_inputs(ctx, x) = x

# Map the tensor leaves of an input tree; passthrough leaves have no derivative.
_tree_map(g, x::_ADTensor) = g(x)
_tree_map(g, x::Tuple) = map(v -> _tree_map(g, v), x)
_tree_map(g, x::NamedTuple{K}) where {K} = NamedTuple{K}(_tree_map(g, Tuple(x)))
_tree_map(g, x) = nothing

_concat() = ()
_concat(a::Tuple) = a
_concat(a::Tuple, b::Tuple, rest::Tuple...) = _concat((a..., b...), rest...)
_tensor_leaves(x::_ADTensor) = (x,)
_tensor_leaves(x::Union{Tuple,NamedTuple}) = _concat(map(_tensor_leaves, Tuple(x))...)
_tensor_leaves(x) = ()
_array_leaves(x::ADArray) = (x,)
_array_leaves(x::Union{Tuple,NamedTuple}) = _concat(map(_array_leaves, Tuple(x))...)
_array_leaves(x) = ()

function _record(backend::_Backend, f::F, x, contexts) where {F}
    ctx = _new_context(backend)
    inputs = _tree_inputs(ctx, x)
    output = f(inputs, map(DI.unwrap, contexts)...)
    _check_output(ctx, output)
    return ctx, inputs, output
end

_check_output(ctx::EagerContext, y::Union{EagerTensor,EagerScalar}) =
    y.graph === ctx.graph || throw(ArgumentError("output belongs to a different context"))
_check_output(ctx::TracedContext, y::TracedTensor) =
    y.graph === ctx.graph || throw(ArgumentError("output belongs to a different context"))
_check_output(ctx, y) = throw(ArgumentError("function must return a Tenferro tensor or scalar of the differentiated context"))

_is_scalar(y::EagerScalar) = true
_is_scalar(y::_ADTensor) = isempty(y.dims)
_output_size(y::EagerScalar) = ()
_output_size(y::_ADTensor) = y.dims
_value(y::EagerScalar) = y[]
_value(y::_ADTensor) = isempty(y.dims) ? only(Array(y)) : Array(y)

_seed(ctx::EagerContext, value::AbstractArray) = EagerTensor(ctx, value)
_seed(ctx::TracedContext, value::AbstractArray) = TracedTensor(ctx, value)

_derivative_tensor(ctx::EagerContext, ::Type{T}, id, dims) where {T} = EagerTensor{T,length(dims),typeof(ctx.graph)}(ctx.graph, id, dims)
_derivative_tensor(ctx::TracedContext, ::Type{T}, id, dims) where {T} = TracedTensor{T,length(dims),typeof(ctx.graph)}(ctx.graph, id, dims)

_reverse_id(ctx::EagerContext, output, input, seed) =
    RustCall.result_to_exception(_rust_call(ctx.graph, RustCrate.vjp_node, UInt(output.id), UInt(input.id), UInt(seed.id)))
_reverse_id(ctx::TracedContext, output, input, seed) =
    RustCall.result_to_exception(_rust_call(ctx.graph, RustCrate.traced_derivative, UInt(output.id), UInt(input.id), UInt(seed.id), false))
_forward_id(ctx::EagerContext, output, input, seed) =
    RustCall.result_to_exception(_rust_call(ctx.graph, RustCrate.jvp_node, UInt(output.id), UInt(input.id), UInt(seed.id)))
_forward_id(ctx::TracedContext, output, input, seed) =
    RustCall.result_to_exception(_rust_call(ctx.graph, RustCrate.traced_derivative, UInt(output.id), UInt(input.id), UInt(seed.id), true))

# Reverse-mode derivative of `output` with respect to one input leaf.
function _reverse(ctx, output, input, seed)
    id = _reverse_id(ctx, output, input, seed)
    id == -2 && return zeros(eltype(input), input.dims)
    id >= 0 || error("tenferro-rs reverse-mode derivative failed")
    Array(_derivative_tensor(ctx, eltype(input), id, input.dims))
end

# Forward-mode derivative; `nothing` when the output does not depend on `input`.
function _forward(ctx, output, input, seed)
    id = _forward_id(ctx, output, input, seed)
    id == -2 && return nothing
    id >= 0 || error("tenferro-rs forward-mode derivative failed")
    Array(_derivative_tensor(ctx, eltype(output), id, _output_size(output)))
end

function _gradient(backend::_Backend, f::F, x, contexts, want_value::Bool) where {F}
    ctx, inputs, output = _record(backend, f, x, contexts)
    _is_scalar(output) || throw(ArgumentError("gradient requires a scalar output"))
    seed = _seed(ctx, fill(one(eltype(output))))
    value = want_value ? _value(output) : nothing
    return value, _tree_map(input -> _reverse(ctx, output, input, seed), inputs)
end

function _check_seed(output, seed::AbstractArray)
    seed isa ADArray || throw(ArgumentError("seed must be a supported array"))
    size(seed) == _output_size(output) || throw(DimensionMismatch("seed shape must match output shape"))
    eltype(seed) === eltype(output) || throw(ArgumentError("seed element type must match output"))
    return seed
end
function _check_seed(output, seed::Number)
    _is_scalar(output) || throw(ArgumentError("scalar seeds require a scalar output"))
    typeof(seed) === eltype(output) || throw(ArgumentError("seed element type must match output"))
    return fill(seed)
end
_check_seed(output, seed) = throw(ArgumentError("seed must be a supported array or number"))

function _pullback(backend::_Backend, f::F, x, seeds::Tuple, contexts, want_value::Bool) where {F}
    ctx, inputs, output = _record(backend, f, x, contexts)
    grads = map(seeds) do seed
        s = _seed(ctx, _check_seed(output, seed))
        _tree_map(input -> _reverse(ctx, output, input, s), inputs)
    end
    return (want_value ? _value(output) : nothing), grads
end

function _pushforward(backend::_Backend, f::F, x, tangents::Tuple, contexts, want_value::Bool) where {F}
    ctx, inputs, output = _record(backend, f, x, contexts)
    leaves = _tensor_leaves(inputs)
    dims = _output_size(output)
    results = map(tangents) do tangent
        directions = _array_leaves(tangent)
        length(directions) == length(leaves) || throw(ArgumentError("each tangent must match the structure of the input"))
        result = zeros(eltype(output), dims)
        for (input, direction) in zip(leaves, directions)
            size(direction) == input.dims || throw(DimensionMismatch("tangent shape must match input shape"))
            eltype(direction) === eltype(input) || throw(ArgumentError("tangent element type must match input"))
            derivative = _forward(ctx, output, input, _seed(ctx, direction))
            derivative === nothing || (result .+= derivative)
        end
        _is_scalar(output) ? result[] : result
    end
    return (want_value ? _value(output) : nothing), results
end

# DifferentiationInterface entry points -----------------------------------------

for B in (:AutoTenferro, :AutoTenferroTraced)
    @eval begin
        DI.value_and_gradient(f::F, backend::$B, x::_DiffInput, contexts::_Constants) where {F} =
            _gradient(backend, f, x, contexts, true)
        DI.gradient(f::F, backend::$B, x::_DiffInput, contexts::_Constants) where {F} =
            last(_gradient(backend, f, x, contexts, false))
        DI.value_and_pullback(f::F, backend::$B, x::_DiffInput, seeds::Tuple, contexts::_Constants) where {F} =
            _pullback(backend, f, x, seeds, contexts, true)
        DI.pullback(f::F, backend::$B, x::_DiffInput, seeds::Tuple, contexts::_Constants) where {F} =
            last(_pullback(backend, f, x, seeds, contexts, false))
        DI.value_and_pushforward(f::F, backend::$B, x::_DiffInput, tangents::Tuple, contexts::_Constants) where {F} =
            _pushforward(backend, f, x, tangents, contexts, true)
        DI.pushforward(f::F, backend::$B, x::_DiffInput, tangents::Tuple, contexts::_Constants) where {F} =
            last(_pushforward(backend, f, x, tangents, contexts, false))
    end
end
