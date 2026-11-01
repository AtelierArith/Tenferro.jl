const _TAPE_KEY = :Tenferro_chainrules_tape

function _apply_rule(op, args::Tuple, forward)
    tape = get(task_local_storage(), _TAPE_KEY, nothing)
    isnothing(tape) && return forward()
    result = task_local_storage(() -> ChainRulesCore.rrule(op, args...), _TAPE_KEY, nothing)
    isnothing(result) && error("No ChainRulesCore.rrule is defined for $op")
    output, back = result
    push!(tape, (output, args, back))
    return output
end

function _trace(f, inputs)
    tape = []
    output = task_local_storage(() -> f(inputs), _TAPE_KEY, tape)
    return output, tape
end

function _reverse(tape, inputs, output, seed)
    cotangents = Dict{Int64, Any}(output.id => seed)
    for (node, args, back) in Iterators.reverse(tape)
        cotangent = get(cotangents, node.id, ZeroTangent())
        cotangent isa AbstractZero && continue
        parts = back(cotangent)
        for (arg, delta) in zip(args, Base.tail(parts))
            delta = unthunk(delta)
            delta isa AbstractZero && continue
            if haskey(cotangents, arg.id)
                cotangents[arg.id] = cotangents[arg.id] + delta
            else
                cotangents[arg.id] = delta
            end
        end
    end
    return map(inputs) do input
        value = get(cotangents, input.id, ZeroTangent())
        value isa AbstractZero ? zeros(eltype(input), input.dims) : Array(value)
    end
end

const ADArray = AbstractArray{<:ADElement}
const ADArrayInputs = Tuple{Vararg{ADArray}}

function _record_inputs(xs::ADArrayInputs)
    isempty(xs) && throw(ArgumentError("at least one input array is required"))
    ctx = EagerContext()
    return ctx.graph, map(x -> EagerTensor(ctx, x), xs)
end

function DifferentiationInterface.gradient(f::F, ::AutoTenferro, xs::ADArrayInputs) where {F}
    graph, inputs = _record_inputs(xs)
    output, tape = _trace(f, inputs)
    output isa EagerScalar || throw(ArgumentError("AutoTenferro gradient requires a scalar output"))
    output.graph === graph || throw(ArgumentError("output belongs to a different runtime"))
    return _reverse(tape, inputs, output, one(eltype(output)))
end

function DifferentiationInterface.gradient(f::F, backend::AutoTenferro, x::ADArray) where {F}
    return only(DifferentiationInterface.gradient(xs -> f(only(xs)), backend, (x,)))
end

function DifferentiationInterface.pullback(f::F, ::AutoTenferro, xs::ADArrayInputs, seeds::Tuple{Vararg{ADArray}}) where {F}
    graph, inputs = _record_inputs(xs)
    output, tape = _trace(f, inputs)
    output isa EagerTensor || throw(ArgumentError("AutoTenferro pullback requires an array output"))
    output.graph === graph || throw(ArgumentError("output belongs to a different runtime"))
    return map(seeds) do seed
        size(seed) == output.dims || throw(DimensionMismatch("seed shape must match output shape"))
        eltype(seed) === eltype(output) || throw(ArgumentError("seed element type must match output"))
        _reverse(tape, inputs, output, EagerTensor(EagerContext(graph), seed))
    end
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
