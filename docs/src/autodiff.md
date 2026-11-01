# Automatic differentiation

`AutoTenferro()` is a DifferentiationInterface backend. It records supported operations in tenferro-rs and uses its JVP and VJP transformations directly. It does not consult ChainRules `rrule` or `frule` methods. Julia 1.13 or later is required.

## Scalar-output gradients

```jldoctest
julia> using Tenferro, DifferentiationInterface

julia> x = [1.0 2.0; 3.0 4.0];

julia> f(x) = sum(x * x);

julia> gradient(f, AutoTenferro(), x) ≈ [7.0 11.0; 9.0 13.0]
true
```

## Multiple inputs and matrix outputs

DifferentiationInterface accepts a single active input. Pass a tuple to differentiate multiple matrices. A pullback accepts a tuple of output cotangent seeds and returns one tuple of input cotangents per seed:

```jldoctest
julia> using Tenferro, DifferentiationInterface

julia> f(x, y) = x * y;

julia> x = [1.0 2.0; 3.0 4.0]; y = [2.0 0.0; 0.0 3.0];

julia> seed = ones(2, 2);

julia> ((dx, dy),) = pullback(args -> f(args...), AutoTenferro(), (x, y), (seed,));

julia> dx ≈ seed * y' && dy ≈ x' * seed
true

julia> gx, gy = gradient(args -> sum(f(args...)), AutoTenferro(), (x, y));

julia> gx ≈ seed * y' && gy ≈ x' * seed
true
```

## Pushforwards

```jldoctest
julia> using Tenferro, DifferentiationInterface

julia> x = [1.0 2.0; 3.0 4.0]; dx = ones(2, 2);

julia> (dy,) = pushforward(a -> a * a, AutoTenferro(), x, (dx,));

julia> dy ≈ dx * x + x * dx
true
```

For tuple inputs, each tangent has the same tuple structure as the input. Multiple tangent directions can be passed in the outer tuple.

## Current scope

The backend accepts CPU arrays with `Float32`, `Float64`, `ComplexF32` or `ComplexF64` elements. Gradient requires a scalar result from a supported reduction. Pullbacks accept scalar or array output seeds, and pushforwards accept tangents matching the input structure, shapes and element types. Addition, subtraction and `sum` support arbitrary ranks, while `*` currently requires matrices. Eager `sum` currently fails on empty arrays. Functions must consist of supported Tenferro operations; defining an `rrule` for an arbitrary Julia function does not make it traceable. Unsupported operations raise errors. Broadcasting, prepared calls and higher-order derivatives are not yet supported.

## Optional ChainRules rules

Loading `ChainRulesCore` together with Tenferro activates the `TenferroChainRulesCoreExt` package extension. It provides first-order `rrule`s for `+`, `-`, matrix `*`, and `sum` on `EagerTensor`; pullbacks return `EagerTensor` cotangents. These rules are separate from `AutoTenferro` and do not guarantee interoperability with a particular AD engine.

## Traced graphs

`TracedTensor` builds a tenferro-rs computation graph. Construct tensors in the same `TracedContext`; converting a result to `Array` compiles and executes its graph on the CPU. `sum` produces a zero-dimensional traced tensor, whose result is read with `Array(result)[]`.

```jldoctest
julia> using Tenferro, DifferentiationInterface

julia> ctx = TracedContext();

julia> x = TracedTensor(ctx, [1.0 2.0; 3.0 4.0]);

julia> Array(sum(x * x))[]
54.0

julia> gradient(x -> sum(x * x), AutoTenferroTraced(), [1.0 2.0; 3.0 4.0]) ≈ [7.0 11.0; 9.0 13.0]
true
```

`AutoTenferroTraced()` uses tenferro-rs graph transformations, through a separate traced context. It supports scalar gradients for an array or a tuple of arrays; traced pullbacks, JVPs, prepared execution and symbolic inputs are not yet exposed. Supported data is CPU arrays of `Float32`, `Float64`, `ComplexF32` or `ComplexF64`; operations are `+`, `-`, matrix `*`, and `sum`. Each `Array` conversion compiles and executes the graph anew.

## Complex differentiation convention

Both backends use tenferro-rs's Hermitian real-inner-product convention: for a scalar holomorphic map with derivative `f′`, the pullback maps a complex cotangent `ȳ` to `ȳ * conj(f′)`. A complex scalar `gradient` here means the VJP with seed `one(eltype(output))`; for example, `sum(A * B)` produces `ones * B'` and `A' * ones`. This describes the cotangent representation and is not a claim that a complex-valued function has an ordinary real-valued gradient. See [tenferro-rs complex autodiff](https://github.com/tensor4all/tenferro-rs/blob/main/docs/guides/complex-ad.md) for the convention. Non-real cotangent seeds are supported by eager array-output pullbacks.
