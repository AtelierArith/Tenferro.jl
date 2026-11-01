# Tenferro.jl

An experimental Julia interface to the local `extern/tenferro-rs` checkout,
using RustCall.jl and DifferentiationInterface.jl.

Julia 1.13 and a Rust toolchain (Rust 1.96 or newer) are required. Clone this repository with `git clone --recurse-submodules https://github.com/AtelierArith/Tenferro.jl.git`, or run `git submodule update --init --recursive` after cloning. The pinned `extern/tenferro-rs` submodule provides the Rust dependencies. Run `julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Tenferro")'`,
and run tests with `julia --project -e 'using Pkg; Pkg.test()'`. The first load
compiles the Rust dependencies if the build cache is missing. The Rust crate
lives in `deps/rust`; `deps/build.jl` can warm RustCall’s build cache. FiniteDiff is used in tests for numerical comparisons.

`EagerTensor` provides a Julia handle for a tenferro-rs eager tensor. Tensors
used together must share a context. Currently CPU tensors of the seven built-in tenferro-rs element types,
`+`, `-`, matrix `*`, and `sum` are supported. Binary operations have
ChainRulesCore reverse rules backed by tenferro-rs VJPs.

```julia
ctx = EagerContext()
a = EagerTensor(ctx, [1.0 2.0; 3.0 4.0])
b = EagerTensor(ctx, ones(2, 2))
Array(a + b)
```

The reverse rules take `EagerTensor` cotangents. They are first-order rules;
the dynamic tensor wrapper remains to be implemented.

`AutoTenferro()` is a separate backend that differentiates with tenferro-rs,
without Zygote. It currently accepts CPU arrays of `Float32`, `Float64`, `ComplexF32` or `ComplexF64` and supports
matrix multiplication, addition, subtraction, and reduction to a scalar.
Its reverse pass composes the ChainRulesCore `rrule` pullbacks registered by
these operations; binary rules call tenferro-rs VJPs:

```julia
using Tenferro, DifferentiationInterface
x = [1.0 2.0; 3.0 4.0]
f(x) = sum(x * x)
grad = DifferentiationInterface.gradient(f, AutoTenferro(), x)
# [7.0 11.0; 9.0 13.0]
```

For multiple active inputs, pass a tuple as the single input to
DifferentiationInterface. Matrix outputs can be differentiated with a pullback:

```julia
f(x, y) = x * y
x = [1.0 2.0; 3.0 4.0]
y = [2.0 0.0; 0.0 3.0]
seed = ones(2, 2)
((dx, dy),) = DifferentiationInterface.pullback(
    args -> f(args...), AutoTenferro(), (x, y), (seed,)
)
# dx == seed * y'; dy == x' * seed

dx, dy = DifferentiationInterface.gradient(
    args -> sum(f(args...)), AutoTenferro(), (x, y)
)
```

Each pullback seed must match the output array’s shape and element type. The
outer result tuple contains one input-gradient tuple per seed.

`AutoTenferro` is experimental: arbitrary Julia functions, broadcasting,
other numeric element types, prepared calls, forward mode, and higher derivatives are not
yet supported. The function must return the result of a supported reduction;
unsupported operations are not silently differentiated by another backend.
Only operations integrated with the eager tape are recorded; an arbitrary Julia
function with an `rrule` is not automatically intercepted.


Run the test suite with `Pkg.test(; test_args=["--jobs=2"])`. Test files run
through ParallelTestRunner; package checks use Aqua and JET.

To build the DocumenterVitepress documentation locally:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=docs docs/make.jl
```

Serve `docs/build/1` over HTTP to preview the generated site.

`TypedTensor` supports CPU arrays with element types `Float32`, `Float64`,
`Int32`, `Int64`, `Bool`, `ComplexF32`, and `ComplexF64`. Eager and traced automatic
differentiation supports the floating-point and complex types; arbitrary
`Number` subtypes are not implicitly converted. Addition, subtraction and
`sum` support arbitrary array ranks, while `*` currently requires matrices.
