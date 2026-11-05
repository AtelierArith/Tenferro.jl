const ADArray = AbstractArray{<:ADElement}
const ADArrayInputs = Tuple{Vararg{ADArray}}

function _record_inputs(xs::ADArrayInputs)
    isempty(xs) && throw(ArgumentError("at least one input array is required"))
    ctx = EagerContext()
    return ctx.graph, map(x -> EagerTensor(ctx, x), xs)
end

function _eager_output(f, xs)
    graph, inputs = _record_inputs(xs)
    output = f(inputs)
    output isa Union{EagerTensor,EagerScalar} || throw(ArgumentError("function must return a Tenferro tensor or scalar"))
    output.graph === graph || throw(ArgumentError("output belongs to a different runtime"))
    return graph, inputs, output
end

function _vjp(graph, inputs, output, seed)
    return map(inputs) do input
        id = RustCall.result_to_exception(_rust_call(graph, RustCrate.vjp_node, UInt(output.id), UInt(input.id), UInt(seed.id)))
        id == -2 && return zeros(eltype(input), input.dims)
        id >= 0 || error("tenferro-rs VJP failed")
        Array(EagerTensor{eltype(input),length(input.dims),typeof(graph)}(graph, id, input.dims))
    end
end

function DifferentiationInterface.gradient(f::F, ::AutoTenferro, xs::ADArrayInputs) where {F}
    graph, inputs, output = _eager_output(f, xs)
    output isa EagerScalar || throw(ArgumentError("AutoTenferro gradient requires a scalar output"))
    seed = EagerTensor(EagerContext(graph), fill(one(eltype(output))))
    return _vjp(graph, inputs, output, seed)
end

function DifferentiationInterface.gradient(f::F, backend::AutoTenferro, x::ADArray) where {F}
    return only(DifferentiationInterface.gradient(xs -> f(only(xs)), backend, (x,)))
end

DifferentiationInterface.pullback(f::F, ::AutoTenferro, x::ADArray, ::Tuple{}) where {F} = ()
DifferentiationInterface.pullback(f::F, ::AutoTenferro, x::ADArrayInputs, ::Tuple{}) where {F} = ()

function DifferentiationInterface.pullback(f::F, ::AutoTenferro, xs::ADArrayInputs, seeds::Tuple{Vararg{ADArray}}) where {F}
    graph, inputs, output = _eager_output(f, xs)
    output isa EagerTensor || throw(ArgumentError("AutoTenferro pullback requires an array output"))
    return map(seeds) do seed
        size(seed) == output.dims || throw(DimensionMismatch("seed shape must match output shape"))
        eltype(seed) === eltype(output) || throw(ArgumentError("seed element type must match output"))
        _vjp(graph, inputs, output, EagerTensor(EagerContext(graph), seed))
    end
end

function DifferentiationInterface.pullback(f::F, backend::AutoTenferro, x::ADArray, seeds::Tuple{Vararg{ADArray}}) where {F}
    return map(only, DifferentiationInterface.pullback(xs -> f(only(xs)), backend, (x,), seeds))
end

function DifferentiationInterface.pullback(f::F, ::AutoTenferro, xs::ADArrayInputs, seeds::Tuple{Vararg{Number}}) where {F}
    graph, inputs, output = _eager_output(f, xs)
    output isa EagerScalar || throw(ArgumentError("scalar seeds require a scalar output"))
    return map(seeds) do seed
        typeof(seed) === eltype(output) || throw(ArgumentError("seed element type must match output"))
        _vjp(graph, inputs, output, EagerTensor(EagerContext(graph), fill(seed)))
    end
end

function DifferentiationInterface.pullback(f::F, backend::AutoTenferro, x::ADArray, seeds::Tuple{Vararg{Number}}) where {F}
    return map(only, DifferentiationInterface.pullback(xs -> f(only(xs)), backend, (x,), seeds))
end

function DifferentiationInterface.pushforward(f::F, ::AutoTenferro, xs::ADArrayInputs, tangents::Tuple) where {F}
    graph, inputs, output = _eager_output(f, xs)
    dims = output isa EagerScalar ? () : output.dims
    return map(tangents) do tangent
        tangent isa Tuple && length(tangent) == length(inputs) || throw(ArgumentError("each tangent must match the input tuple"))
        result = zeros(eltype(output), dims)
        for (input, direction) in zip(inputs, tangent)
            direction isa ADArray || throw(ArgumentError("tangent must be a supported array"))
            size(direction) == input.dims || throw(DimensionMismatch("tangent shape must match input shape"))
            eltype(direction) === eltype(input) || throw(ArgumentError("tangent element type must match input"))
            seed = EagerTensor(EagerContext(graph), direction)
            id = RustCall.result_to_exception(_rust_call(graph, RustCrate.jvp_node, UInt(output.id), UInt(input.id), UInt(seed.id)))
            id == -2 && continue
            id >= 0 || error("tenferro-rs JVP failed")
            result .+= Array(EagerTensor{eltype(output),length(dims),typeof(graph)}(graph, id, dims))
        end
        output isa EagerScalar ? result[] : result
    end
end

function DifferentiationInterface.pushforward(f::F, backend::AutoTenferro, x::ADArray, tangents::Tuple{Vararg{ADArray}}) where {F}
    return DifferentiationInterface.pushforward(xs -> f(only(xs)), backend, (x,), map(t -> (t,), tangents))
end

function DifferentiationInterface.gradient(f::F, ::AutoTenferroTraced, xs::ADArrayInputs) where {F}
    isempty(xs) && throw(ArgumentError("at least one input array is required"))
    ctx = TracedContext()
    inputs = map(x -> TracedTensor(ctx, x), xs)
    output = f(inputs)
    output isa TracedTensor && isempty(output.dims) || throw(ArgumentError("AutoTenferroTraced requires a scalar output"))
    output.graph === ctx.graph || throw(ArgumentError("output belongs to a different context"))
    return map(inputs) do input
        id = RustCall.result_to_exception(_rust_call(ctx.graph, RustCrate.traced_grad, UInt(output.id), UInt(input.id)))
        id == -2 && return zeros(eltype(input), input.dims)
        id >= 0 || error("tenferro-rs traced gradient failed")
        Array(TracedTensor{eltype(input),length(input.dims),typeof(ctx.graph)}(ctx.graph, id, input.dims))
    end
end

function DifferentiationInterface.gradient(f::F, backend::AutoTenferroTraced, x::ADArray) where {F}
    return only(DifferentiationInterface.gradient(xs -> f(only(xs)), backend, (x,)))
end

function _traced_output(f, xs)
    isempty(xs) && throw(ArgumentError("at least one input array is required"))
    ctx = TracedContext()
    inputs = map(x -> TracedTensor(ctx, x), xs)
    output = f(inputs)
    output isa TracedTensor || throw(ArgumentError("function must return a Tenferro traced tensor"))
    output.graph === ctx.graph || throw(ArgumentError("output belongs to a different context"))
    return ctx.graph, inputs, output
end

function _traced_vjp(graph, inputs, output, seed)
    map(inputs) do input
        id = RustCall.result_to_exception(_rust_call(graph, RustCrate.traced_derivative, UInt(output.id), UInt(input.id), UInt(seed.id), false))
        id == -2 && return zeros(eltype(input), input.dims)
        id >= 0 || error("tenferro-rs traced VJP failed")
        Array(TracedTensor{eltype(input),length(input.dims),typeof(graph)}(graph, id, input.dims))
    end
end

DifferentiationInterface.pullback(f::F, ::AutoTenferroTraced, x::ADArray, ::Tuple{}) where {F} = ()
DifferentiationInterface.pullback(f::F, ::AutoTenferroTraced, x::ADArrayInputs, ::Tuple{}) where {F} = ()

function DifferentiationInterface.pullback(f::F, ::AutoTenferroTraced, xs::ADArrayInputs, seeds::Tuple{Vararg{ADArray}}) where {F}
    graph, inputs, output = _traced_output(f, xs)
    return map(seeds) do seed
        size(seed) == output.dims || throw(DimensionMismatch("seed shape must match output shape"))
        eltype(seed) === eltype(output) || throw(ArgumentError("seed element type must match output"))
        _traced_vjp(graph, inputs, output, TracedTensor(TracedContext(graph), seed))
    end
end

function DifferentiationInterface.pullback(f::F, backend::AutoTenferroTraced, x::ADArray, seeds::Tuple{Vararg{ADArray}}) where {F}
    return map(only, DifferentiationInterface.pullback(xs -> f(only(xs)), backend, (x,), seeds))
end

function DifferentiationInterface.pullback(f::F, ::AutoTenferroTraced, xs::ADArrayInputs, seeds::Tuple{Vararg{Number}}) where {F}
    graph, inputs, output = _traced_output(f, xs)
    isempty(output.dims) || throw(ArgumentError("scalar seeds require a scalar output"))
    return map(seeds) do seed
        typeof(seed) === eltype(output) || throw(ArgumentError("seed element type must match output"))
        _traced_vjp(graph, inputs, output, TracedTensor(TracedContext(graph), fill(seed)))
    end
end

function DifferentiationInterface.pullback(f::F, backend::AutoTenferroTraced, x::ADArray, seeds::Tuple{Vararg{Number}}) where {F}
    return map(only, DifferentiationInterface.pullback(xs -> f(only(xs)), backend, (x,), seeds))
end

function DifferentiationInterface.pushforward(f::F, ::AutoTenferroTraced, xs::ADArrayInputs, tangents::Tuple) where {F}
    graph, inputs, output = _traced_output(f, xs)
    dims = output.dims
    return map(tangents) do tangent
        tangent isa Tuple && length(tangent) == length(inputs) || throw(ArgumentError("each tangent must match the input tuple"))
        result = zeros(eltype(output), dims)
        for (input, direction) in zip(inputs, tangent)
            direction isa ADArray || throw(ArgumentError("tangent must be a supported array"))
            size(direction) == input.dims || throw(DimensionMismatch("tangent shape must match input shape"))
            eltype(direction) === eltype(input) || throw(ArgumentError("tangent element type must match input"))
            seed = TracedTensor(TracedContext(graph), direction)
            id = RustCall.result_to_exception(_rust_call(graph, RustCrate.traced_derivative, UInt(output.id), UInt(input.id), UInt(seed.id), true))
            id == -2 && continue
            id >= 0 || error("tenferro-rs JVP failed")
            result .+= Array(TracedTensor{eltype(output),length(dims),typeof(graph)}(graph, id, dims))
        end
        isempty(output.dims) ? result[] : result
    end
end

function DifferentiationInterface.pushforward(f::F, backend::AutoTenferroTraced, x::ADArray, tangents::Tuple{Vararg{ADArray}}) where {F}
    return DifferentiationInterface.pushforward(xs -> f(only(xs)), backend, (x,), map(t -> (t,), tangents))
end
