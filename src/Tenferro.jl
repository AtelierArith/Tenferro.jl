module Tenferro

using RustCall
using ChainRulesCore
using ADTypes: AbstractADType
import DifferentiationInterface

export TypedTensor, AutoTenferro, EagerContext, EagerTensor, AutoTenferroTraced, TracedContext, TracedTensor

const RustCrate = @rust_crate joinpath(@__DIR__, "..", "deps", "rust")

include("ffi.jl")
include("types.jl")
include("operators.jl")
include("autodiff.jl")
include("TenferroChainRules.jl")

end
