# Getting started

## Setup

The current development checkout requires Julia 1.13 or later and a Rust toolchain. The local `extern/tenferro-rs` checkout must be present because `deps/rust/Cargo.toml` references it by path. From the package root, instantiate and build:

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Tenferro")'
```

The first build compiles Rust dependencies. `deps/build.jl` warms RustCall's cache; the package can rebuild its Rust bindings when the Rust sources change.

## Eager tensors

A context owns the eager execution state. Operands in an operation must share the same context. Constructing an `EagerTensor` copies an array of a supported built-in element type into tenferro-rs; `Array` copies the result back to Julia.

```jldoctest
julia> using Tenferro

julia> ctx = EagerContext();

julia> a = EagerTensor(ctx, [1.0 2.0; 3.0 4.0]);

julia> b = EagerTensor(ctx, ones(2, 2));

julia> Array(a + b) == [2.0 3.0; 4.0 5.0]
true
```

The current tensor surface supports `+`, `-`, matrix `*`, and `sum`. Operands must have compatible shapes. Addition, subtraction and `sum` support arbitrary ranks; matrix multiplication requires two matrices. Operation support depends on element type: for example, arithmetic on `Bool` is not generally defined. GPU tensors and additional operations are not yet exposed.

## Julia arrays and typed tensors

`TypedTensor` wraps a CPU tenferro-rs typed tensor and implements Julia's `AbstractArray` interface. Construct it from an array or view of any dimensionality; construction copies values in column-major order. `Array(tensor)` makes an independent Julia array. Indexing and mutation operate on the Rust-owned tensor.

```jldoctest
julia> using Tenferro

julia> x = TypedTensor(reshape(Float32[1, 2, 3, 4], 2, 2));

julia> x[1, 2] = 9;

julia> Array(x)
2×2 Matrix{Float32}:
 1.0  9.0
 2.0  4.0
```

Supported element types are `Float32`, `Float64`, `Int32`, `Int64`, `Bool`, `ComplexF32` and `ComplexF64`. Eager and traced constructors accept these `TypedTensor` values by copying. Generic Julia array operations can fall back to Julia implementations and may return ordinary arrays.

## Supported element types

The seven built-in scalar types defined by tenferro-rs map to Julia as follows:

| tenferro-rs dtype | Julia element type | Differentiable |
| --- | --- | --- |
| `F32` | `Float32` | Yes |
| `F64` | `Float64` | Yes |
| `I32` | `Int32` | No |
| `I64` | `Int64` | No |
| `Bool` | `Bool` | No |
| `C32` (`Complex32`) | `ComplexF32` | Yes |
| `C64` (`Complex64`) | `ComplexF64` | Yes |

`TypedTensor`, `EagerTensor` and `TracedTensor` accept all seven built-in types on the CPU. `AutoTenferro` and `AutoTenferroTraced` accept only the four differentiable types in the table. Acceptance as tensor storage does not imply every operation works for every dtype; in particular, `Bool` does not support numeric arithmetic. Operands of a binary operation must currently have matching element types.

The tenferro-rs data model can also represent externally defined scalar types, but its built-in operation catalog does not automatically implement operations for them. Tenferro.jl does not currently bridge arbitrary external scalar types or Julia `Number` subtypes such as `BigFloat`; it does not silently convert them. See the [tenferro-rs dtype definitions](https://github.com/tensor4all/tenferro-rs/blob/main/crates/tenferro-tensor-core/src/lib.rs) and [operation capability policy](https://github.com/tensor4all/tenferro-rs/blob/main/crates/tenferro-tensor/src/capability.rs).

## Context lifetime and concurrency

`EagerContext` and `TracedContext` own their Rust graph and retain its nodes while the context is open. When a workload is finished, call `close(ctx)` to invalidate its tensors and release the Julia reference to the Rust graph; the graph is freed by RustCall when its wrapper is garbage-collected. Closing a context twice is harmless, and any subsequent operation on its tensors throws `ArgumentError`. Contexts also remain alive while their tensors are reachable, so dropping only the context variable does not invalidate those tensors.

Operations on a shared context are serialized with a lock, as are operations on each `TypedTensor` storage. Individual graph calls can therefore be used from multiple Julia tasks, though a sequence of calls is not atomic with respect to other tasks or `close`. All contexts of a process execute on one shared tenferro-rs CPU engine, so calls from different contexts also take turns; see [Performance](performance.md) for its thread setting. Nodes are not yet released individually: for repeated independent workloads, close the context and create a fresh one to bound retained graph state.
