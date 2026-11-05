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

See [Operation contracts](operations.md) for type, axis, ownership, domain,
and decomposition-result details, including differences from Julia Base.

Scalar reductions can be composed with arithmetic and supported scalar math functions, for example `exp(sum(x))`.

The backend accepts CPU arrays with `Float32`, `Float64`, `ComplexF32` or `ComplexF64` elements. Gradient requires a scalar result. Pullbacks and pushforwards accept seeds and tangents matching the corresponding values.

Supported eager and traced operations include:

- Elementwise broadcast of `+`, `-`, `*`, `/`, `^`, `rem`, `min`, `max`, `clamp`, `abs`, `sign`, `conj`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `expm1`, and `log1p`; unary `-`, and scalar multiplication and division. Mixed floating-point and complex tensor operands are promoted according to Julia rules when the resulting type is supported.
- `sum`, `prod`, `maximum`, `minimum`, and `Statistics.mean`, with or without `dims`.
- `reshape`, `permutedims`, vector or matrix `transpose` and `adjoint`, `reverse`, `cat` along one existing dimension, and indexing with one integer, nonempty positive-step integer range, or `:` per dimension. One-dimensional tensors also accept vectors of integer indices.
- Matrix multiplication and matrix–vector `*`, `\` (including tall rectangular least-squares problems), `inv`, `pinv`, `det`, `norm`, thin `svd`, `qr`, `lu`, `eigen`, `cholesky`, `tril`, `triu`, `LowerTriangular`/`UpperTriangular` solves, and `diag` for nonempty diagonals. Decompositions return Tenferro-specific result types, rather than Julia standard factorization types. `svd` exposes `U`/`S`/`Vt`/`V`, `qr` exposes `Q`/`R`, `lu` exposes `P`/`L`/`U`, `eigen` exposes `values`/`vectors`, and `cholesky` exposes `L`/`U`. For real inputs `eigen(A)` returns complex-valued eigenvalues and eigenvectors; `eigen(Hermitian(A))` returns real eigenvalues and uses the triangle specified by `Hermitian`. By default `eigen` materializes eigenvalues to determine Julia-compatible sorting; `sortby=nothing` preserves the backend order without that sorting. Derivatives at repeated eigenvalues may be undefined.

`cholesky(A)` materializes the input to check Hermitian symmetry; `cholesky(Hermitian(A))` uses the specified triangle. For `rem.(x, y)`, a Tenferro.jl Rust extension supplies JVP and VJP rules: away from quotient discontinuities, the derivatives are `1` with respect to `x` and `-div(x, y)` (quotient rounded toward zero) with respect to `y`. The derivative is undefined at nonzero integer quotient boundaries and when the divisor is zero. Complex `sign` is differentiated as `z / abs(z)` for nonzero `z`; its derivative is undefined at zero. Singular solves and inverses report tenferro-rs errors; numerical factorization failures are propagated. Derivatives of decompositions can be undefined or ill-conditioned at singular or repeated values, and available derivatives depend on tenferro-rs. Full SVD, arbitrary broadcast functions, general indexing, prepared calls, and higher derivatives are not yet supported. For empty inputs, reductions use Julia’s empty-array semantics: `sum` and `prod` use their identities, `mean` yields NaN where an empty slice is reduced, and `maximum`/`minimum` throw when an output slice has no elements. Empty `sum` and `prod` have empty input gradients. Functions must use supported Tenferro operations; defining an `rrule` for an arbitrary Julia function does not make it traceable.

For real tensors, `log` and `sqrt` at negative inputs, and `log1p` below -1, return NaN from the backend; Julia scalar functions throw `DomainError` for those inputs. At the real boundary, `log(0)` and `log1p(-1)` produce negative infinity, and `sqrt(0)` produces zero. Derivatives at singularities and outside the real domain should not be assumed finite. Complex functions use the backend's complex arithmetic and derivatives; branch cuts and singularities remain subject to their mathematical restrictions.

Loading SpecialFunctions.jl enables `erf.(x)` for `Float32` and `Float64` eager and traced tensors through a package extension. The Rust backend supports its first-order AD on those real types; complex `erf` is not supported by this extension.

## Broadcast and element types

Binary broadcast follows Julia's axis alignment: missing trailing dimensions are singleton dimensions, and each pair of extents must agree or include a singleton. Scalars and constant Julia arrays participate in this expansion. Incompatible shapes throw `DimensionMismatch`; tensors in different contexts are rejected. Supported mixed floating-point and complex types use `promote_type`, including scalar types (a `Float64` scalar can promote a `Float32` tensor). Gradients are returned in each active input's original element type and shape; broadcast axes accumulate their cotangents. Integer and Boolean storage is available, but `AutoTenferro` and `AutoTenferroTraced` accept only the four floating-point AD input types. GPU arrays and GPU execution are outside this interface's supported contract.

For same-type `Int32`/`Int64` tensors, binary broadcast `+`, `-`, `*`, `rem`, `max`, `min`, and integer powers execute without AD and retain the integer dtype. Integer `./` promotes operands to Julia's floating-point division result type. Boolean tensors support storage, conversion, and `./` with floating-point results; the other binary arithmetic broadcasts listed here are rejected. Integer overflow and invalid integer division/powers follow backend restrictions rather than floating-point AD semantics.

| Julia operation | Backend operation / contract |
| --- | --- |
| `.+`, `.-`, unary `-` | add, subtract, negate |
| `.*`, `./`, `.^` | elementwise multiply, divide, power; matrix `*` is separate |
| `rem.` / `.%` | remainder; real inputs, derivatives away from quotient discontinuities |
| `max.`, `min.`, `clamp.` | real comparisons; binary ties split the cotangent equally |
| `abs.`, `conj.`, `sign.` | magnitude (real output for complex input), conjugate, sign |
| `exp.`, `log.`, `sqrt.`, `sin.`, `cos.`, `tanh.`, `expm1.`, `log1p.` | elementwise math for the four AD types; domain restrictions described above |

For `maximum` and `minimum` reductions, equal extrema split the cotangent equally among all matching entries. Binary `max`/`min` split a tie equally between the two operands; `clamp` inherits this convention from its composition of `max` and `min`. These are backend conventions at non-differentiable points.

At zero, real `abs` and `sign`, complex magnitude and complex sign have non-smooth or undefined derivatives; backend choices there do not imply mathematical differentiability. Likewise, division by zero, power singularities, and comparison ties should not be treated as smooth points.

## Shape, indexing, and storage contract

Tensor constructors copy Julia arrays, including views, into Rust-owned storage in Julia column-major order. `Array(tensor)` returns an independent Julia array. Mutating the original array or the returned array does not mutate the tensor. Each tensor holds its graph state, which retains its Rust nodes and their storage; keeping a result alive also keeps the graph alive even if its context and input Julia variables are collected. Explicitly closing a context invalidates tensors sharing that context. The graph serializes Rust calls with a lock. Shape operations can share Rust storage internally, but the tensor API does not expose mutable Julia views of it; no alias to a Julia input buffer is retained.

Dimensions and indices in the Julia API are one-based; the wrapper converts axes and indices to zero-based Rust values. `reshape` follows Julia's column-major element order; `permutedims` reorders axes, `transpose` does not conjugate, and `adjoint` does. `cat` accepts one existing dimension. Indexing accepts one integer, nonempty positive-step range or colon per dimension; integer indices drop that dimension. A vector additionally accepts integer index vectors (including repeated indices) and Boolean masks. Empty or negative-step ranges and general multidimensional array indexing are not supported. These restrictions apply to eager and traced tensors, not Julia arrays generally.

## Linear algebra contract

The factorization wrappers accept matrices, not batches of matrices. `svd` returns thin factors with `k = min(m,n)`: `U` is `m × k`, `S` has length `k`, and `Vt` is `k × n`, with `A ≈ U * Diagonal(S) * Vt`. `svd(...; full=true)` is rejected for both eager and traced tensors: the backend's full SVD is value-only and its AD is unavailable. `qr` returns thin `Q` and `R` with the same `k` convention and `A ≈ Q * R`. `lu` exposes the permutation matrix `P`, lower factor `L`, and upper factor `U`, satisfying `P * A ≈ L * U`. The factorization objects are Tenferro-specific, not subtypes of Julia's standard factorization types; their other Julia APIs and keyword arguments are not generally implemented.

Singular vectors and QR/eigenvector phases or signs are not uniquely determined; compare reconstructions rather than individual factors. Pivoting and repeated or zero singular/eigenvalues can make derivatives discontinuous, undefined, or ill-conditioned. Julia and the backend can choose different rank thresholds or pivots; rank-deficient least-squares and pseudoinverse compatibility is not guaranteed. Matrix and vector solves require compatible dimensions; tall rectangular systems use the backend least-squares implementation. Underdetermined wide systems are rejected by the backend. These operations run on CPU for the four AD element types listed above; no batched or GPU interface is exposed here.


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

`AutoTenferroTraced()` uses tenferro-rs graph transformations, through a separate traced context. It supports scalar gradients, pullbacks and pushforwards for an array or a tuple of arrays, with the same seed and tangent shape/type requirements as the eager backend. Prepared execution and symbolic inputs are not yet exposed. Supported data is CPU arrays of `Float32`, `Float64`, `ComplexF32` or `ComplexF64`; operations include those listed above. Each `Array` conversion compiles and executes the graph anew.

## Complex differentiation convention

Both backends use tenferro-rs's Hermitian real-inner-product convention: for a scalar holomorphic map with derivative `f′`, the pullback maps a complex cotangent `ȳ` to `ȳ * conj(f′)`. A complex scalar `gradient` here means the VJP with seed `one(eltype(output))`; for example, `sum(A * B)` produces `ones * B'` and `A' * ones`. This describes the cotangent representation and is not a claim that a complex-valued function has an ordinary real-valued gradient. See [tenferro-rs complex autodiff](https://github.com/tensor4all/tenferro-rs/blob/main/docs/guides/complex-ad.md) for the convention. Non-real cotangent seeds are supported by eager and traced array-output pullbacks.
