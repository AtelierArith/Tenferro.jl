module Tenferro

using RustCall
using Statistics
using LinearAlgebra
import ADTypes
using ADTypes: AbstractADType
import DifferentiationInterface

public stopgrad, constant, context, DifferentiableElement, num_threads, set_num_threads

export TypedTensor, AutoTenferro, EagerContext, EagerTensor, EagerScalar, AutoTenferroTraced, TracedContext, TracedTensor

@rust_crate joinpath(@__DIR__, "..", "deps", "rust") submodule="RustCrate"

include("ffi.jl")
include("types.jl")
include("operators.jl")
include("interface.jl")
include("nnops.jl")
include("arrayops.jl")
include("autodiff.jl")

end
