module TenferroChainRulesCoreExt

using ChainRulesCore
using Tenferro: EagerTensor, EagerContext
using Tenferro: _rust_call, RustCrate
using RustCall

# ChainRules reverse rules for Tenferro eager operations.

function ChainRulesCore.rrule(::typeof(sum), a::EagerTensor)
    output = sum(a)
    function pullback(seed)
        seed = unthunk(seed)
        seed isa AbstractZero && return NoTangent(), ZeroTangent()
        seed isa Number || throw(ArgumentError("sum cotangent must be a number"))
        return NoTangent(), EagerTensor(EagerContext(a.graph), fill(convert(eltype(a), seed), a.dims))
    end
    return output, pullback
end

function ChainRulesCore.rrule(op::Union{typeof(+),typeof(-),typeof(*)}, a::EagerTensor, b::EagerTensor)
    output = op(a, b)
    function rule_input(x::EagerTensor{T,N}) where {T,N}
        id = RustCall.result_to_exception(_rust_call(x.graph, RustCrate.rule_input, UInt(x.id)))
        id >= 0 || error("tenferro-rs could not construct pullback input")
        return EagerTensor{T,N,typeof(x.graph)}(x.graph, id, x.dims)
    end
    rule_a, rule_b = rule_input(a), rule_input(b)
    rule_output = op(rule_a, rule_b)
    function pullback(cotangent)
        cotangent = unthunk(cotangent)
        cotangent isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        cotangent isa EagerTensor || throw(ArgumentError("expected an EagerTensor cotangent"))
        cotangent.dims == output.dims || throw(DimensionMismatch("seed shape must match output"))
        eltype(cotangent) === eltype(output) || throw(ArgumentError("seed element type must match output"))
        seed = cotangent.graph === a.graph ? cotangent : EagerTensor(EagerContext(a.graph), Array(cotangent))
        function vjp(x::EagerTensor{T,N}) where {T,N}
            id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.vjp_node, UInt(rule_output.id), UInt(x.id), UInt(seed.id)))
            id == -2 && return ZeroTangent()
            id >= 0 || error("tenferro-rs pullback failed")
            EagerTensor{T,N,typeof(a.graph)}(a.graph, id, x.dims)
        end
        return NoTangent(), vjp(rule_a), vjp(rule_b)
    end
    return output, pullback
end

end
