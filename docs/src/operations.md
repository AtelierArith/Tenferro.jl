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

`+`, `*`, `max` and `min` broadcast over more than two arguments as Julia's
left fold, for example `max.(x, y, 0)`, and `muladd.(a, b, c)` and
`fma.(a, b, c)` are `a .* b .+ c`. Partially applied functions with a scalar,
such as `(>(0)).(x)` or `Base.Fix1(-, 1).(x)`, broadcast as the binary
operation. `zero.(x)` and `one.(x)` are constant tensors. `similar` throws an
`ArgumentError`: tensors are immutable.

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

## Element-type casts and constants

`S.(x)` converts a tensor to any tensor element type `S` (`Float32`, `Float64`,
`ComplexF32`, `ComplexF64`, `Int32`, `Int64`, `Bool`) with the tenferro-rs
`cast`, for example `Float32.(x)` of a `Float64` tensor, `ComplexF64.(x)` of a
real tensor, or `Float32.(x .> 0)` of a mask; `float.(x)` casts to the
floating-point type of `eltype(x)`. `convert(EagerTensor{S}, x)` (and likewise
for `TracedTensor`) is the same conversion and returns `x` itself when it
already has the requested type. Casts between floating-point and complex types
are differentiable: cotangents are cast back to the input type. Casts to or from
integer and `Bool` types carry no derivative.

Conversions follow Julia's `S(v)` for each element:

- Conversions that are exact for every value of the source type always
  succeed: to floating-point and complex types from real types (rounding to
  nearest, as `Float32(v)` does), from `Bool`, and from `Int32` to `Int64`.
- Conversions that depend on the values (a floating-point or complex value to an
  integer type, `Int64` to `Int32`, anything but `Bool` to `Bool`, and complex to
  real) check every element of an eager tensor and throw Julia's
  `InexactError` when one is not exactly representable: `Int64.(x)` throws for
  `2.7`, `NaN` or `2.0^63`, `Bool.(x)` for anything but 0 and 1, `Int32.(x)` for
  `2^40`, and `Float64.(z)` for a nonzero imaginary part. The check costs one
  read of the tensor back to the host.
- On a `TracedTensor` the values are only computed when the graph is evaluated,
  so these value-dependent conversions throw an `ArgumentError` instead. Use an
  explicit rounding (`trunc.(Int64, x)`, `round.(Int64, x)`, `floor.(Int64, x)`
  or `ceil.(Int64, x)`), `x .!= 0` for `Bool`, `x .% Int32` to wrap integers,
  or `real.(z)` and `imag.(z)` for complex tensors.

`trunc.(x)`, `floor.(x)`, `ceil.(x)` and `round.(x)` (ties to even, as in Julia;
`round.(x, mode)` also accepts `RoundNearestTiesAway`, `RoundNearestTiesUp`,
`RoundToZero`, `RoundFromZero`, `RoundUp` and `RoundDown`) round real
floating-point tensors and keep their element type, `NaN`, infinities and the
sign of zero. Their derivative is zero (for magnitudes of `2^62` and above,
where every float is already an integer, it is zero up to rounding in the
accumulated cotangent). Integer tensors round to themselves.
They are composed from casts, comparisons and selections, so they cost more than
one elementwise operation.

`trunc.(S, x)`, `floor.(S, x)`, `ceil.(S, x)`, `round.(S, x)` and
`round.(S, x, mode)` for `S` in `Int32`, `Int64` and `Bool` round and then
convert. On an eager tensor a value whose rounding is out of range for `S`, or
is `NaN` or infinite, throws the same `InexactError` as Julia. On a traced
tensor the backend cast reports such values as an error when the graph is
evaluated; a `Bool` target is rejected with an `ArgumentError`. An integer
tensor converts exactly, as `S.(x)` does. `x .% S` for an integer tensor `x` and
`S` in `Int32` or `Int64` wraps like Julia's `rem(v, S)` on both kinds of
tensor. Mixed-type arithmetic promotes with the raw cast, which only widens.

`Tenferro.context(x)` returns the `EagerContext` or `TracedContext` that owns
`x`, and `Tenferro.constant(x, array)` copies a Julia array or number into it as
a constant, without derivatives, keeping its element type. Inside a function
differentiated by `AutoTenferro()` or `AutoTenferroTraced()`, this is how to
create tensors that combine with the traced input.

## Comparisons and selection

`.==`, `.!=`, `.<`, `.<=`, `.>` and `.>=` compare tensors, scalars and constant
Julia arrays with the broadcast and promotion rules above and return `Bool`
tensors. Comparisons are not differentiable: no cotangent flows through them.
`.!` negates a `Bool` tensor. `ifelse.(mask, a, b)` selects elementwise; the
mask may be a `Bool` tensor, a `Bool` array or a `Bool`, and the branches may
be tensors, scalars or constant arrays of a common promoted type. Cotangents
flow into the selected branch only. `abs2.(x)` and `inv.(x)` are composed from
the supported arithmetic operations.

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
Loading SpecialFunctions.jl enables `erf.(x)` for real tensors through a
package extension.

## Broadcast fallback

Only the functions listed in this guide are lowered to tenferro-rs. Broadcasting any other function over a tensor, including an anonymous function or a closure, throws an `ArgumentError` that names the function; write its body as separate dotted operations. `(f ∘ g).(x...)` is expanded to `f.(g.(x...))`, and `identity.(x)` returns `x`. An `EagerScalar` broadcasts with tensors, numbers and constant arrays.

## Neural-network operations

These operations are available after `using NNlib` (package extension `TenferroNNlibExt`), for eager and traced tensors, and are differentiable by both backends.

| NNlib function | Implementation and derivative at kinks |
|---|---|
| `sigmoid`/`σ`, `sigmoid_fast`, `softplus`, `swish`, `gelu`/`gelu_tanh`, `gelu_erf` | tenferro-rs activation kernels (overflow-free `sigmoid` and `softplus`) |
| `relu` | native kernel `x > 0 ? x : 0` (a tenferro-rs extension, with derivative `x > 0`: `0` at the kink) |
| `relu6`, `trelu` | `select`; derivative `0` at every kink |
| `leakyrelu(x, a=0.01)` | `select`; derivative `a` at `0` |
| `elu(x, α=1)`, `celu(x, α=1)`, `selu` | `select` with `expm1`; `elu'(0) == celu'(0) == 1`, `selu'(0) == λα` |
| `hardσ`, `hardtanh`, `softshrink(x, λ=0.5)` | `select`; derivative `0` at the kinks |
| `hardswish` | `x .* hardσ.(x)` with NNlib's derivative `x/3 + 1/2` on `[-3, 3]` |
| `logσ`, `softsign`, `mish`, `lisht`, `tanhshrink`, `logcosh`, `tanh_fast`, `gelu_sigmoid`, `quick_gelu` | composed from the smooth operations above |

The activations accept `Float32` and `Float64` tensors, and eager scalars. `leakyrelu`, `elu`, `celu`, `softshrink` and `trelu` also accept their parameter as a constant real number, as in `leakyrelu.(x, 0.2)`. `rrelu` (random) is not provided. `NNlib.fast_act(f, x)` returns `f` for tensors. Unlike NNlib's `tanh_fast` and `sigmoid_fast`, the tensor versions are not approximations, so `Float32` results can differ from NNlib's in the last few bits.

`softmax(x; dims=1)`, `logsoftmax(x; dims=1)` and `logsumexp(x; dims=:)` accept an integer, a tuple of dimensions or `:`. For a single dimension, softmax and logsoftmax use the tenferro-rs kernels; otherwise they are composed from `maximum`, `exp`, `sum` and `log`, shifted by the maximum along `dims`.

`batched_mul(A, B)` (`A ⊠ B`) multiplies the matrices `A[:, :, k]` and `B[:, :, k]` through tenferro-rs `dot_general` with a batch dimension. It follows NNlib: a 3-D operand with batch size one is reused for every batch, a matrix operand multiplies every batch, and higher-rank operands with equal trailing dimensions are reshaped to one batch dimension. Either operand may be a constant Julia array. `batched_transpose` and `batched_adjoint` return permuted (and, for complex tensors, conjugated) tensors, not lazy wrappers.

`pad_zeros(x, pad; dims=:)` and `pad_constant(x, pad, val=0; dims=:)` follow NNlib's padding arguments: an integer, one amount per padded dimension, a `(low, high)` amount pair per padded dimension, or a tuple of `(low, high)` pairs. They use tenferro-rs `pad`; a nonzero `val` adds a constant frame. Negative padding (cropping) is not supported.

### Convolution, pooling and upsampling

tenferro-rs has no convolution or `reduce_window` primitive. For `Float32` and `Float64` tensors, Tenferro's bridge adds native window kernels as a tenferro-rs extension family: `im2col` (copy every window into a column matrix) and its transpose `col2im`, window sums and their transpose, and first-maximum pooling with its tangent and cotangent maps. Every derivative of a member is a member again, so pushforwards and pullbacks of any order run on these kernels, eagerly and traced. Other element types (complex tensors) compose the same operations from `pad`, `slice`, `concatenate`, reduction and `select` kernels. All of them follow NNlib's WHCN layout, `(spatial..., channels, batch)` with one to three spatial dimensions, and NNlib's argument conventions.

| NNlib function | Implementation |
|---|---|
| `conv(x, w; stride, pad, dilation, flipped, groups)`, `conv(x, w, ::DenseConvDims)` | `im2col` of `x` into a `(kernel..., C_in, out..., batch)` matrix, then one `dot_general` over kernel offsets and input channels; groups are a `dot_general` batch dimension |
| `depthwiseconv(x, w; ...)`, `depthwiseconv(x, w, ::DepthwiseConvDims)` | grouped `conv` with `groups = C_in` |
| `∇conv_data(dy, w, cdims)`, `∇depthwiseconv_data` | transposed convolution: `conv` of `dy` with interior (stride) padding and the channel-swapped kernel reversed once more |
| `∇conv_filter(x, dy, cdims)`, `∇depthwiseconv_filter` | windows of `x` contracted with `dy` |
| `maxpool(x, k; pad, stride, ceil_mode)`, `maxpool(x, ::PoolDims)` | running maximum over the kernel offsets in column-major order, padding counting as `-Inf`, replaced only by a strictly larger entry, so each window's gradient goes to its first maximum in kernel order, as in NNlib's `∇maxpool` |
| `meanpool(x, k; pad, stride, count_include_pad, ceil_mode)`, `meanpool(x, ::PoolDims)` | window sums divided by NNlib's per-window count: all window positions inside the padded input, or only those inside the input with `count_include_pad=false` |
| `lpnormpool(x, p, k; pad, stride, ceil_mode)`, `lpnormpool(x, ::PoolDims; p)` | `(∑ xᵢ^p)^(1/p)` over each window (padding contributes nothing) |
| `upsample_nearest(x, scales)`, `upsample_nearest(x; size)` | `broadcast_to` of a reshaped tensor |
| `upsample_linear`, `upsample_bilinear`, `upsample_trilinear` (scale or `size`, both `align_corners`) | one pair of axis `gather`s with constant interpolation weights per resized dimension (NNlib's source positions) |
| `pixel_shuffle(x, r)` | `reshape` and `permutedims` |

`PoolDims`, `DenseConvDims` and `DepthwiseConvDims` accept tensors in place of arrays. Either operand of `conv`, `depthwiseconv`, `∇conv_data` and `∇conv_filter` may be a constant Julia array. `maximum` and `minimum` accept `init` (Lux's `GlobalMaxPool` calls `maximum(x; dims, init=-Inf)`), so Lux's global and adaptive pooling layers work. Anti-aliased bilinear upsampling, `upsample_bicubic` and complex inputs are not provided.

`NNlib.gather(src, idx)` (and `gather(src, I, J, ...)`) accepts integers,
integer tuples and `CartesianIndex`es as indices: one axis gather of `src` with
the indexed trailing dimensions flattened. `NNlib.scatter(op, src, idx; init,
dstsize)` supports `op` in `+`, `-` and `Statistics.mean`, built on the additive
tenferro-rs `scatter`; repeated indices accumulate. Other operators (`max`,
`min`, `*`, `/`) throw an `ArgumentError`. Indices are checked on the host;
out-of-range indices throw a `BoundsError` instead of being clamped.

## Reductions

`sum`, `prod`, `maximum`, `minimum`, and `Statistics.mean` accept `dims`;
`prod`, `maximum` and `minimum` also accept `init`, combined with the result by
one more `*`, `max` or `min`.
`Statistics.var` and `Statistics.std` accept `dims`, `corrected` and a
precomputed `mean`; they are composed from `mean`, `abs2` and `sum`.
`sum(f, x)`, `prod(f, x)`, `maximum(f, x)`, `minimum(f, x)` and
`Statistics.mean(f, x)` reduce `f.(x)`, with the same keywords; `f` must be a
supported elementwise function such as `abs2`.
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

On zero-length reduced axes, the Julia wrapper supplies the sum/product
identities, `mean` returns `NaN`, and `maximum`/`minimum` throw as in Julia.
The backend identity extension is tracked in
[tenferro-rs#2453](https://github.com/tensor4all/tenferro-rs/issues/2453).
A zero-length *non-reduced* dimension gives an empty output with the expected
shape. An empty input is distinct from an empty tuple of reduction axes.
The dtype and AD guarantees in this section apply to floating/complex types;
integer widening (for example Julia's `sum(::Vector{Int32})`) is not a supported
Julia-compatible contract.

`Bool` tensors reduce like Julia's: `sum(mask)` and `count(mask)` (also
`count(f, x)`) return `Int64` counts, `mean(mask)` and the mean of an integer
tensor are `Float64`, and `any(mask)`, `all(mask)`, `any(f, x)` and `all(f, x)`
return `Bool`, all with optional `dims`. Without `dims`, `any` and `all` return a
zero-dimensional `Bool` tensor (an eager scalar or a zero-dimensional traced
tensor), not a Julia `Bool`, so convert it with `only(any(mask))` or
`Bool(any(mask))` before using it in `if` or `&&`.

## Shapes, indexing, and ownership

`reshape` follows Julia column-major element order. `permutedims` accepts a
permutation of one-based axes. A vector's `transpose` or `adjoint` has shape
`(1,n)`; a matrix swaps its two axes. `adjoint` also conjugates values and its
pullback conjugates the seed; `transpose` does not conjugate. Higher-rank
`transpose` is rejected; use `permutedims` instead.

`reverse` accepts one dimension or a tuple of dimensions. `cat` accepts one
positive dimension; lower-rank operands get trailing singleton dimensions, so
`cat(a, b; dims=3)` of matrices or `hcat` of vectors works as in Julia.
Every other axis must have equal extents; floating-point/complex types are
promoted. `vcat` and `hcat` of tensors are `cat` along dimensions 1 and 2.
Constant Julia arrays may be mixed with tensors in `cat`, `vcat` and `hcat`
(the first tensor must be among the first three arguments), for example
`hcat(x, ones(3, 1))`. `vec(x)` flattens to a vector, and `repeat(x, counts...)`
and `repeat(x; inner, outer)` follow Base with positive counts, as a reshape and
`broadcast_to` whose cotangents sum over the repeated entries.
Block concatenation tuples are not exposed. `dropdims`, `selectdim`, and
`eachslice` (returning a Julia array of tensor slices, with `drop`) follow Base.

Indexing supports one integer, nonempty positive-step integer range, `:`,
nonempty integer vector, or Boolean mask per dimension, for example
`W[:, [1, 3, 3]]`. Integer dimensions are dropped. Integer vectors gather along
their dimension; repeated indices accumulate their cotangents. Empty ranges
and negative steps are explicitly rejected; `A[:]` flattens. Linear indexing
of matrices, Cartesian index arrays, `view`, and `setindex!` are not exposed. `tril`, `triu`, and
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
