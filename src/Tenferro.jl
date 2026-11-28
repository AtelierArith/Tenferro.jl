module Tenferro

using RustCall
using Statistics
using LinearAlgebra
import ADTypes
using ADTypes: AbstractADType
import DifferentiationInterface

public stopgrad, constant, context, DifferentiableElement, num_threads, set_num_threads, dense, logitcrossentropy

export TypedTensor, AutoTenferro, EagerContext, EagerTensor, EagerScalar, AutoTenferroTraced, TracedContext, TracedTensor

@rust_crate joinpath(@__DIR__, "..", "deps", "rust") submodule="RustCrate"

include("ffi.jl")
include("types.jl")
include("operators.jl")
include("interface.jl")
include("nnops.jl")
include("arrayops.jl")
include("autodiff.jl")

# The native dense kernels multiply with Julia's BLAS through libblastrampoline's ILP64
# `sgemm`/`dgemm`; `on = false` switches them to tenferro-rs's built-in GEMM (faer).
# Returns whether BLAS was in use before.
function _use_blas_gemm(on::Bool)
    sgemm = on ? UInt(cglobal((LinearAlgebra.BLAS.@blasfunc(sgemm_), LinearAlgebra.BLAS.libblastrampoline))) : UInt(0)
    dgemm = on ? UInt(cglobal((LinearAlgebra.BLAS.@blasfunc(dgemm_), LinearAlgebra.BLAS.libblastrampoline))) : UInt(0)
    return RustCrate.set_blas_gemm(sgemm, dgemm) != 0
end

__init__() = (_use_blas_gemm(true); nothing)

end
