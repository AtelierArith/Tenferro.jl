# Tenferro.jl

Tenferro.jl is an experimental Julia interface to [tenferro-rs](https://github.com/tensor4all/tenferro-rs). It uses RustCall.jl to run tensor operations and exposes tenferro-rs automatic differentiation through DifferentiationInterface.jl.

The bundled tenferro-rs submodule is pinned to v0.9.0. The Rust bridge uses its
semantic-program AD APIs and the threadteam 0.3 CPU dependency stack.

## Setup

Julia 1.13 or later and a Rust toolchain are required. Initialize the tenferro-rs submodule, then instantiate and build the package:

```sh
git submodule update --init --recursive
julia --project -e 'using Pkg; Pkg.instantiate(; workspace=true); Pkg.build("Tenferro")'
```

The first load compiles the Rust dependencies if the build cache is missing. The Rust crate is in `deps/rust`; `deps/build.jl` can warm RustCall's build cache.

## Automatic differentiation

`AutoTenferro()` differentiates functions composed of supported Tenferro tensor operations. It uses tenferro-rs JVP and VJP transformations directly; it does not require Zygote or ChainRules rules.

```julia
using Tenferro, DifferentiationInterface

x = [1.0 2.0; 3.0 4.0]
f(x) = sum(x * x)

gradient(f, AutoTenferro(), x)
# [7.0 11.0; 9.0 13.0]
```

`pushforward` computes a Jacobian-vector product for each input tangent. `pullback` computes a vector-Jacobian product for each output cotangent. Both take an outer tuple of directions:

```julia
dx = ones(2, 2)
(dy,) = pushforward(a -> a * a, AutoTenferro(), x, (dx,))
# dy ≈ dx * x + x * dx

seed = ones(2, 2)
(dx_back,) = pullback(a -> a * a, AutoTenferro(), x, (seed,))
# dx_back ≈ seed * x' + x' * seed
```

For multiple active inputs, pass a tuple as the single DifferentiationInterface input. Each tangent or returned cotangent follows that input structure:

```julia
f(x, y) = x * y
y = [2.0 0.0; 0.0 3.0]
((dx, dy),) = pullback(args -> f(args...), AutoTenferro(), (x, y), (seed,))
# dx ≈ seed * y'; dy ≈ x' * seed

gx, gy = gradient(args -> sum(f(args...)), AutoTenferro(), (x, y))
```

Seeds and tangents must match the corresponding value's shape and element type. Complex cotangents use the real Hermitian inner-product convention; a complex scalar gradient is a VJP with seed `one(eltype(output))`. See the [autodiff guide](docs/src/autodiff.md) for details.

Supported differentiable inputs are CPU arrays of `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`. Supported operations include arithmetic broadcast, elementwise mathematical functions, reductions, shape and slicing operations, and selected linear algebra. See the [autodiff guide](docs/src/autodiff.md) for the current list and limitations. Eager `sum` currently fails on empty arrays.

Arbitrary Julia functions are supported only when their tensor operations are supported by Tenferro. General Julia broadcast functions, prepared calls, and higher-order derivatives are not yet supported. Defining an `rrule` does not make an arbitrary Julia function traceable by `AutoTenferro()`.

## Tensor types

Detailed axis, dtype, domain, ownership, and factorization contracts are in the
[operation guide](docs/src/operations.md).

- `TypedTensor(array)` wraps CPU tensor storage and implements Julia's `AbstractArray` interface. Conversion to and from Julia arrays copies the data. Supported element types are `Float32`, `Float64`, `Int32`, `Int64`, `Bool`, `ComplexF32`, and `ComplexF64`; arbitrary `Number` subtypes are not implicitly converted.
- `EagerTensor(ctx, array)` executes operations eagerly in a shared `EagerContext` and records them for tenferro-rs AD. Tensors used in an operation must belong to the same context.
- `TracedTensor(ctx, array)` builds a graph in a `TracedContext`. `AutoTenferroTraced()` exposes scalar gradients using traced graph transformations. Traced pushforwards, pullbacks, prepared execution, and symbolic inputs are not yet exposed.

```julia
ctx = EagerContext()
a = EagerTensor(ctx, [1.0 2.0; 3.0 4.0])
b = EagerTensor(ctx, ones(2, 2))
Array(a + b)
```

## Optional ChainRules extension

Loading `ChainRulesCore` together with Tenferro activates the `TenferroChainRulesCoreExt` package extension. It provides first-order `rrule`s for `+`, `-`, matrix `*`, and `sum` on `EagerTensor`:

```julia
using Tenferro, ChainRulesCore

ctx = EagerContext()
x = EagerTensor(ctx, [1.0 2.0; 3.0 4.0])
output, back = rrule(*, x, x)
_, dx1, dx2 = back(EagerTensor(ctx, ones(2, 2)))
```

The pullbacks return `EagerTensor` cotangents. Compatibility with a particular external AD engine is not guaranteed. This extension is separate from Tenferro's own `AutoTenferro()` backend.

## Tests and documentation

Run the test suite with `julia --project -e 'using Pkg; Pkg.test(; test_args=["--jobs=2"])'`. Tests use ParallelTestRunner, Aqua, JET, and FiniteDiff.

Build the DocumenterVitepress documentation locally with:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=docs docs/make.jl
```

Serve `docs/build/1` over HTTP to preview the generated site.
