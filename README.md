# Tenferro.jl

Tenferro.jl is an experimental Julia interface to [tenferro-rs](https://github.com/tensor4all/tenferro-rs). It uses RustCall.jl to run tensor operations and exposes tenferro-rs automatic differentiation through DifferentiationInterface.jl.

The bundled tenferro-rs submodule is pinned to the 0.12.0 release commit
(`d48982d3`, tagged `v0.12.0`). The Rust bridge uses the 0.12 one-call-grammar
APIs, its semantic-program AD APIs, and the single faer-backed CPU dependency
stack (`threadteam` 0.5).

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

`value_and_gradient`, `value_and_pullback` and `value_and_pushforward` also return the primal value, DifferentiationInterface `Constant` arguments are passed through undifferentiated, and the active input may be a nested `Tuple`/`NamedTuple` parameter tree (gradients keep its structure):

```julia
ps = (weight = [1.0 2.0; 3.0 4.0], bias = [0.5, -1.0])
loss(p, x) = sum(abs2.(p.weight * x .+ p.bias))
value, g = value_and_gradient(loss, AutoTenferro(), ps, Constant([1.0, 2.0]))
# g.weight, g.bias
```

Use `Tenferro.stopgrad(x)` (or `ChainRulesCore.ignore_derivatives(x)`) to block gradient flow. Scalar results are read with `s[]`, `only(s)` or `Float64(s)`.

Seeds and tangents must match the corresponding value's shape and element type. Complex cotangents use the real Hermitian inner-product convention; a complex scalar gradient is a VJP with seed `one(eltype(output))`. See the [autodiff guide](docs/src/autodiff.md) for details.

Supported differentiable inputs are CPU arrays of `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`. Supported operations include arithmetic broadcast, comparisons and `ifelse` selection, elementwise mathematical functions, reductions (including `var`/`std`), shape, concatenation, slicing and integer-vector indexing operations, and selected linear algebra. See the [autodiff guide](docs/src/autodiff.md) for the current list and limitations.

Element types are cast with broadcasting, as in `Float32.(x)` or `Float64.(x .> 0)` (differentiable between floating-point and complex types). Inside a differentiated function, `Tenferro.constant(x, array)` creates a constant tensor in the context of `x` (`Tenferro.context(x)` returns that context), and plain arrays can be mixed with tensors in arithmetic and in `cat`/`hcat`/`vcat`. `vec`, `repeat`, `sum(f, x; dims)`-style mapped reductions, `count`/`any`/`all` and sums of `Bool` tensors, n-ary `max.`/`min.`/`+`/`*`, and `muladd.`/`fma.` are also supported.

Arbitrary Julia functions are supported only when their tensor operations are supported by Tenferro. Broadcasting any other function, including an anonymous function, throws an `ArgumentError` naming it; `(f ∘ g).(x...)` is expanded to `f.(g.(x...))`. Prepared calls and higher-order derivatives are not yet supported. Defining an `rrule` does not make an arbitrary Julia function traceable by `AutoTenferro()`.

## Tensor types

Detailed axis, dtype, domain, ownership, and factorization contracts are in the
[operation guide](docs/src/operations.md).

- `TypedTensor(array)` wraps CPU tensor storage and implements Julia's `AbstractArray` interface. Conversion to and from Julia arrays copies the data. Supported element types are `Float32`, `Float64`, `Int32`, `Int64`, `Bool`, `ComplexF32`, and `ComplexF64`; arbitrary `Number` subtypes are not implicitly converted.
- `EagerTensor(ctx, array)` executes operations eagerly in a shared `EagerContext` and records them for tenferro-rs AD. Tensors used in an operation must belong to the same context.
- `TracedTensor(ctx, array)` builds a graph in a `TracedContext`. `AutoTenferroTraced()` exposes gradients, pushforwards and pullbacks using traced graph transformations. Prepared execution and symbolic inputs are not yet exposed.

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

## Optional NNlib extension

Loading NNlib.jl together with Tenferro activates the `TenferroNNlibExt` package extension, so the NNlib operations that Lux.jl layers use work on `EagerTensor` and `TracedTensor` and can be differentiated by `AutoTenferro()` and `AutoTenferroTraced()`:

- activation broadcasts such as `relu.(x)`, `gelu.(x)`, `swish.(x)`, `sigmoid.(x)`, `softplus.(x)` and `leakyrelu.(x, 0.2)`, plus `NNlib.fast_act(f, x)`. Derivatives at kinks follow NNlib's ChainRules (for example `relu'(0) == 0`);
- `softmax`, `logsoftmax` and `logsumexp` with `dims`;
- `batched_mul` (`⊠`), `batched_transpose` and `batched_adjoint`;
- `pad_zeros` and `pad_constant`;
- `conv` (stride, asymmetric padding, dilation, groups, `flipped`), `depthwiseconv`, `∇conv_data` (transposed convolution) and `∇conv_filter` with 1-3 spatial dimensions;
- `maxpool` (gradients go to the first maximum of each window, like NNlib), `meanpool` and `lpnormpool` with `PoolDims` or keywords, including what Lux's global and adaptive pooling layers call;
- `upsample_nearest`, `upsample_linear`/`upsample_bilinear`/`upsample_trilinear` (both `align_corners`) and `pixel_shuffle`;
- `gather` and `scatter` (`+`, `-`, `mean`).

```julia
using Tenferro, NNlib, DifferentiationInterface

layer(p, x) = sum(softmax(relu.(p.W * x .+ p.b); dims=1) .* [1.0, 2.0])
ps = (W = [1.0 -2.0; 0.5 1.5], b = [0.1, -0.3])
value, g = value_and_gradient(layer, AutoTenferro(), ps, Constant([1.0, 2.0]))
```

```julia
x = randn(Float32, 28, 28, 1, 8)
cnn(w) = sum(maxpool(relu.(conv(x, w; pad=1)), (2, 2)))
g = gradient(cnn, AutoTenferroTraced(), randn(Float32, 3, 3, 1, 4))
```

Normalization layers and dropout are not yet provided.

## Optional Functors extension

Loading Functors.jl activates `TenferroFunctorsExt`, which declares `EagerTensor`, `TracedTensor` and eager scalars as Functors.jl leaves, so `fmap` and `fleaves` over parameter trees (as Lux and Optimisers.jl use them) treat each tensor as one value.

## Tests and documentation

Run the test suite with `julia --project -e 'using Pkg; Pkg.test(; test_args=["--jobs=2"])'`. Tests use ParallelTestRunner, Aqua, JET, FiniteDiff, NNlib, and Functors.

Build the DocumenterVitepress documentation locally with:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=docs docs/make.jl
```

Serve `docs/build/1` over HTTP to preview the generated site.

CI publishes the documentation to the `gh-pages` branch after tests and the
documentation build pass on `main`. The [development documentation](https://atelierarith.github.io/Tenferro.jl/dev/)
tracks `main`; pull requests only build the documentation. In **Settings → Pages**,
select **Deploy from a branch**, branch **gh-pages**, and folder **/ (root)**.
Deployment authenticates with the repository's `DOCUMENTER_KEY` secret and its
write-enabled deploy key so each push triggers a Pages update.
