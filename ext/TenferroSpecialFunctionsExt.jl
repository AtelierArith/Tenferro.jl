module TenferroSpecialFunctionsExt

using SpecialFunctions: erf
using Tenferro: EagerTensor, TracedTensor, RustCrate, _rust_call
using RustCall

function Base.Broadcast.broadcasted(::typeof(erf), a::EagerTensor{T,N}) where {T<:Union{Float32,Float64},N}
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.unary, UInt(a.id), Int32(12)))
    return EagerTensor{T,N,typeof(a.graph)}(a.graph, id, a.dims)
end
function Base.Broadcast.broadcasted(::typeof(erf), a::TracedTensor{T,N}) where {T<:Union{Float32,Float64},N}
    id = RustCall.result_to_exception(_rust_call(a.graph, RustCrate.traced_unary, UInt(a.id), Int32(12)))
    return TracedTensor{T,N,typeof(a.graph)}(a.graph, id, a.dims)
end

end
