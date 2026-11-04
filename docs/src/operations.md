# Operation contracts

This page describes the Julia interface. An operation in tenferro-rs does not
automatically imply the same operation is exposed by Tenferro.jl.

## Types and devices

`AutoTenferro()` and `AutoTenferroTraced()` accept CPU arrays of `Float32`,
`Float64`, `ComplexF32`, and `ComplexF64`. Integer and `Bool` arrays are storage
and forward-computation types, not differentiable inputs. GPU construction,
device transfer, and GPU execution are not exposed by this Julia package.
Converting a GPU array through the `AbstractArray` constructor would copy to
the host; it does not select a GPU backend.

`TypedTensor` provides storage and Julia's `AbstractArray` fallback operations.
Those fallback operations are not recorded for Tenferro AD. Use `EagerTensor`
or `TracedTensor` for the Rust operation and AD surfaces described here.

## Arithmetic and broadcasting

`A + B` and `A - B` require equal shapes. `A * B` is matrix multiplication
(or a matrix–vector product), whereas `A .* B` is elementwise multiplication.
Dot operations support Julia's leading-axis broadcast convention: missing
trailing dimensions have extent one. The Julia adapter explicitly expands
operands before calling Rust, so Rust's own implicit broadcast convention is
not used. Cotangents of expanded axes are summed back to each input's shape.

Binary dot operations are `+`, `-`, `*`, `/`, `^`, `rem` (also `.%`), `min`,
and `max`. Scalars and constant Julia arrays can appear on either side. Mixed
floating-point and complex operands use Julia's `promote_type`; cotangents are
converted back to each active input's type. Unsupported promoted types are
rejected. Tensors in one operation must share a context.

Same-type `Int32`/`Int64` tensors support forward `+`, `-`, elementwise `*`,
`rem`, `min`, `max`, `abs`, and `sign`. Elementwise `/` converts integer and
Boolean operands to `Float64`, matching Julia rather than Rust's integer
division. Mixed integer tensor types and integer type promotion are not a
supported general interface. Boolean arithmetic is otherwise rejected by Rust.

`clamp.(x, lo, hi)` is composed as `min.(max.(x, lo), hi)` and allows broadcast
bounds. `abs.`, `sign.`, `conj.`, and unary `-` are elementwise operations.
Complex `abs` returns the corresponding real element type. Ordering operations
(`min`, `max`, `clamp`, `minimum`, `maximum`) are restricted to real values.

Derivatives at quotient discontinuities of `rem`, zero of `abs`/`sign`,
power branch points, or equal operands of `min`/`max` are not classical
derivatives. `rem` uses derivatives `1` and `-div(x,y)` away from its
discontinuities; its divisor must be nonzero. Complex `sign` uses the real
derivative of `z / abs(z)` away from zero. Do not use finite differences across
these discontinuities as a gradient reference.

## Elementwise mathematical functions

The eight supported broadcast functions call the corresponding Rust operations:

| Julia | Derivative away from poles and branch cuts |
|---|---|
| `exp.(x)`, `expm1.(x)` | `exp.(x)` |
| `log.(x)` | `1 ./ x` |
| `log1p.(x)` | `1 ./ (1 .+ x)` |
| `sqrt.(x)` | `1 ./ (2 .* sqrt.(x))` |
| `sin.(x)` | `cos.(x)` |
| `cos.(x)` | `-sin.(x)` |
| `tanh.(x)` | `1 .- tanh.(x).^2` |

Complex VJPs conjugate the analytic derivative. Tests compare both precisions
and complex types against Julia values, analytic derivatives, and directional
finite differences away from singularities. `sqrt` at zero, `log` at zero,
and `log1p` at minus one have singular derivatives. Complex branch cuts and
poles are excluded from differentiability claims. Real input is not implicitly
promoted to complex to extend a function's domain.

Real negative arguments of `log` and `sqrt`, and arguments below minus one of
`log1p`, currently produce Rust's IEEE `NaN`, whereas Julia Base throws a
`DomainError`. `log(0)` and `log1p(-1)` produce `-Inf`. Integer and Boolean
inputs to these eight mathematical functions are rejected rather than
implicitly converted to floating-point; explicitly convert the input first.

`exp(A)` is a matrix function and is not provided by these elementwise methods.
`erf` and SpecialFunctions.jl integration are outside the current supported
surface; no SpecialFunctions dependency or extension is installed.

## Reductions

`sum`, `prod`, `maximum`, `minimum`, and `Statistics.mean` accept `dims`.
Without `dims`, they reduce all elements to a scalar. With `dims`, Julia
dimensions are converted from one-based to zero-based Rust axes; reduced
dimensions are restored with extent one. Tuples of axes may be unordered.
Duplicate or nonpositive axes are rejected; axes beyond the input rank are
ignored. An empty tuple of axes returns the input unchanged.

Product derivatives account for zero factors: one zero receives the product
of the other factors, and two or more zeros give zero derivatives. Tied
reduction extrema divide the cotangent equally among the tied elements.
This is a selected subgradient, not differentiability at a tie. It differs
from [ChainRules' extrema rules](https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/array.jl),
which route the seed to the single index selected by `findmin`/`findmax`.
For example, the gradient of `minimum([1,1,2])` here is `[0.5,0.5,0]`,
whereas ChainRules selects `[1,0,0]`. Product zero-factor handling agrees with
the [ChainRules product rule](https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/mapreduce.jl).
These comparisons do not install or invoke ChainRules during Tenferro AD. Complex
`sum`, `prod`, and `mean` are supported; complex extrema are not.

On zero-length reduced axes, both backends reject `sum(...; dims=...)`,
`prod`, `maximum`, and `minimum`; `mean` returns `NaN`. In contrast Julia
returns zero/one for sum/product and errors for extrema. The sum/product
identity extension is tracked in
[tenferro-rs#2453](https://github.com/tensor4all/tenferro-rs/issues/2453).
A zero-length *non-reduced* dimension gives an empty output with the expected
shape. An empty input is distinct from an empty tuple of reduction axes.
The dtype and AD guarantees in this section apply to floating/complex types;
integer widening (for example Julia's `sum(::Vector{Int32})`) and Boolean
reductions are not supported Julia-compatible contracts.

## Shapes, indexing, and ownership

`reshape` follows Julia column-major element order. `permutedims` accepts a
permutation of one-based axes. A vector's `transpose` or `adjoint` has shape
`(1,n)`; a matrix swaps its two axes. `adjoint` also conjugates values and its
pullback conjugates the seed; `transpose` does not conjugate. Higher-rank
`transpose` is rejected; use `permutedims` instead.

`reverse` accepts one dimension or a tuple of dimensions. `cat` accepts one
existing dimension, requires equal extents on every other axis, and promotes
floating-point/complex types. New dimensions and block concatenation tuples
are not exposed.

Indexing supports one integer, nonempty positive-step integer range, or `:`
per dimension. Integer dimensions are dropped. Vectors additionally support
integer index vectors and Boolean masks; repeated indices accumulate their
cotangents. Empty ranges and negative steps are explicitly rejected. General
linear indexing of matrices, Cartesian index arrays, `view`, `setindex!`, and
arbitrary mixed advanced indexing are not exposed. `tril`, `triu`, and
nonempty `diag` use Julia's signed diagonal offsets.

Input construction and `Array(tensor)` each copy data. Raw Julia pointers are
used only during the FFI call under `GC.@preserve`; Rust copies input buffers
before returning. A tensor retains its shared context, so the context variable
need not remain in the caller's scope. Rust may share immutable storage for
shape/view operations, but no mutable Julia alias or write-through view is
returned. Mutating an input Julia array or a materialized result does not
modify a tensor. Explicitly closing a context invalidates every tensor that
shares it. Context accesses are serialized by a Julia lock.

## Linear algebra

The Julia decomposition results are lightweight Tenferro containers. They are
not subtypes of Julia's standard factorization types and do not provide all
their methods or keyword options. For an `m × n` matrix and `k = min(m,n)`:

| Operation | Result contract |
|---|---|
| `svd(A)` | Thin `U: (m,k)`, real `S: (k,)`, `Vt: (k,n)`; `A ≈ U * Diagonal(S) * Vt`; `.V` is `Vt'`; iteration gives `U,S,V` |
| `qr(A)` | Unpivoted thin `Q: (m,k)`, `R: (k,n)`; `A ≈ Q*R` |
| `lu(A)` | `P: (m,m)`, `L: (m,k)`, `U: (k,n)`; `P*A ≈ L*U`; no `.p` or Julia pivot-vector contract |
| `cholesky(A)` | Lower factor `.L`, conjugate transpose `.U`; `A ≈ L*U`; checks Hermitian input |
| `eigen(A)` | `.values`, `.vectors`; general eigenvalues and eigenvectors are complex even for real input |
| `eigen(Hermitian(A,uplo))` | Real eigenvalues; uses only the selected triangle, with real diagonal |
| `A \ B` | Square solve, or least squares for rectangular `A`; accepts vector or matrix right-hand side |
| `LowerTriangular(A) \ B`, `UpperTriangular(A) \ B` | Uses the selected triangle |
| `det`, `inv`, `pinv`, `norm` | Scalar determinant, inverse, pseudoinverse, and Euclidean/Frobenius norm |

Eigenvalues use Julia-compatible sorting by default; `sortby=nothing` retains
backend order. Decomposition signs and complex phases are not fixed to match
Julia; compare reconstructions, invariant quantities, or consistently aligned
factors. Batch dimensions, pivoted QR, full SVD, configurable pseudoinverse
tolerances, and general norm orders are not exposed. Full SVD is explicitly
rejected because Rust does not supply its AD rule.

The adapter propagates numerical failures from Rust. Singular solves/inverses,
nonpositive Cholesky inputs, nonconvergence, changing rank, repeated singular
or eigenvalues, zero norm, and pivot changes may fail or have undefined or
ill-conditioned derivatives. These are not covered by smooth-gradient claims.
The default `pinv` rank threshold belongs to Rust; it is not promised to equal
Julia's default threshold at a numerical rank boundary.
