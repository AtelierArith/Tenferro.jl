module Tenferro

using RustCall
using Statistics
using LinearAlgebra
import ADTypes
using ADTypes: AbstractADType
import DifferentiationInterface

public stopgrad

export TypedTensor, AutoTenferro, EagerContext, EagerTensor, AutoTenferroTraced, TracedContext, TracedTensor

const RustCrate = @rust_crate joinpath(@__DIR__, "..", "deps", "rust")

include("ffi.jl")
include("types.jl")
include("operators.jl")
include("interface.jl")
include("autodiff.jl")

end
