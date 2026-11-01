# Tenferro.jl

An experimental Julia interface to the local `extern/tenferro-rs` checkout,
using RustCall.jl. Its first operation is `Tenferro.multiply` for equally shaped
CPU `Array{Float64}` values. The operation runs on tenferro-rs; a
ChainRulesCore `rrule` calls its eager VJP, allowing Zygote to differentiate it.

```julia
using Tenferro, DifferentiationInterface, ADTypes
using Zygote: Zygote  # Load the selected backend.

x, y = [2.0, 3.0], [4.0, 5.0]
f(a) = sum(Tenferro.multiply(a, y))
dx = DifferentiationInterface.gradient(f, AutoZygote(), x)
# dx == [4.0, 5.0]
```

Julia 1.13 and a Rust toolchain are required. Ensure `extern/tenferro-rs` is
present, run `julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Tenferro")'`,
and run tests with `julia --project -e 'using Pkg; Pkg.test()'`. The first load
compiles the Rust dependencies if the build cache is missing. The Rust crate
lives in `deps/rust`; `deps/build.jl` can warm RustCall’s build cache. Zygote and FiniteDiff are test dependencies.
Applications add the interface and the AD backend they choose to their own
environment. DifferentiationInterface selects a backend; it does not perform
automatic differentiation by itself. The ChainRules `rrule` is used by
compatible backends. `AutoFiniteDiff` is also tested as a numerical comparison;
other AD backends have not yet been verified.

`AutoTenferro()` is a separate backend that differentiates with tenferro-rs,
without Zygote. It currently accepts CPU `Matrix{Float64}` inputs and supports
matrix multiplication and reduction to a scalar:

```julia
using Tenferro, DifferentiationInterface
x = [1.0 2.0; 3.0 4.0]
f(x) = sum(x * x)
grad = DifferentiationInterface.gradient(f, AutoTenferro(), x)
# [7.0 11.0; 9.0 13.0]
```

`AutoTenferro` is experimental: arbitrary Julia functions, broadcasting,
other input types, prepared calls, forward mode, and higher derivatives are not
yet supported. The function must return the result of a supported reduction;
unsupported operations are not silently differentiated by another backend.

The `multiply` integration is an initial integration: inputs and gradients are copied, the Rust
forward computation is rebuilt during each VJP, and only first-order reverse
mode for this operation is supported. It does not yet support GPU tensors,
other dtypes, general tenferro operations, or higher-order derivatives.
# Tenferro.jl
