# Performance

## The shared CPU engine and its threads

All tenferro-rs operations of a process run on one shared CPU execution engine. The engine
keeps its buffer pool, which recycles output allocations, and its kernel plan cache between
calls and across contexts. Calls from different contexts and Julia threads take turns on
it.

By default the engine is serial. To run kernels on a worker pool, set the
`TENFERRO_NUM_THREADS` environment variable before the first tensor operation, or call
`Tenferro.set_num_threads` at any time:

```julia
using Tenferro
Tenferro.set_num_threads(4)   # rebuilds the engine with 4 workers
Tenferro.num_threads()        # 4
```

Changing the thread count rebuilds the engine, so its buffer pool and plan cache start
empty. Tensors and contexts created before the change remain valid. Threaded kernels may
sum in a different order than serial ones, so results can differ in the last bits. The
default stays serial so that results are reproducible and the engine does not compete with
Julia or BLAS threads.

## Reusing derivative programs

All eager contexts share one tenferro-rs eager runtime. Its caches of compiled derivative
programs and AD transforms persist across contexts and DifferentiationInterface calls, and
tenferro-ad bounds them at 128 entries and 64 MiB each. The cache key is the structure of
the recorded computation and the shapes and element types of its inputs, not their values.
A second `gradient` call on the same function with new data of the same shapes therefore
reuses the compiled reverse pass, even when the function is a new closure.

Each DifferentiationInterface call closes its context before it returns, which frees the
tenferro-rs graph at once. Contexts left for Julia's garbage collector would stay registered
in the shared runtime, and every reverse pass scans the tensors of all live contexts, so
steps would get slower until a collection runs.

Gradients, and each seed of a pullback, are computed for all inputs in one reverse pass. On
the traced backend, the value and all gradients are compiled and run as one program. A
`TracedContext` keeps the value of every node it has evaluated until it is closed, so
reading a value again does not recompile.

Traced programs are also prepared only once per structure. Each `TracedContext` is a new
trace, but a training step traces the same computation every time. The shared engine keeps
one tenferro-rs runtime and its 32 most recently prepared programs. A new program with the
same normalized structure and the same input element types and shapes reuses the prepared
one and runs on the new trace's input tensors. Data does not have to match; numbers and
arrays mixed into the computation are inputs too. Each cached program keeps the input
tensors of the trace that prepared it until it is evicted. Changing the thread count clears
the cache together with the engine. On the MNIST MLP step this took `AutoTenferroTraced`
from 1.24 to 0.95–0.97 ms at one thread and from 1.31 to 0.90 ms at eight (median of
interleaved runs; CNN 16.2 → 15.4 ms and 15.1 → 13.8 ms).

Arrays and numbers that are mixed into tensor operations, `Tenferro.constant` values and
`DifferentiationInterface.Constant` data are untracked constants. No gradient is computed
for them, and they do not take part in reverse passes.

## Dense layers and cross-entropy

A dense layer composed from tensor operations is a `dot_general`, a bias broadcast, an add
and an activation, and its derivative adds several more instructions. With a fixed cost per
operation and instruction, a small model spends more time in that bookkeeping than in its
kernels. `Tenferro.dense(σ, W, x, b)` runs `σ.(W * x .+ b)` as one native operation for
`identity` and `relu` with `Float32` or `Float64` operands, and its pullback is one native
operation per layer. `Tenferro.logitcrossentropy(ŷ, y; agg)` does the same for the
softmax cross-entropy of logits. TenferroLux routes `Lux.Dense`, LuxLib's
`fused_dense_bias_activation` and `CrossEntropyLoss(; logits=Val(true))` to them.

The matrix products of the native dense kernels call Julia's BLAS through
libblastrampoline (`sgemm_64_`/`dgemm_64_`), so they use the BLAS library and the
`BLAS.set_num_threads` setting of the Julia session, not Tenferro's worker pool. Tenferro
registers these entry points when it loads; without them the kernels use a built-in GEMM
(faer) on the calling thread.

## Crossing the Julia boundary

`EagerTensor(ctx, x)`, `TracedTensor(ctx, x)` and `Array(t)` copy numeric data with one
`memcpy` in each direction; only `Bool` data is converted element by element. Eager
gradients are copied straight from tenferro-rs's gradient slot into the Julia result,
without wrapping them in a tensor first, and the unit seed of a scalar gradient is made on
the Rust side.

The Rust library keeps recently freed blocks of 32 KiB to 64 MiB, up to 64 blocks and 256
MiB in total, and hands them out again for requests of the same rounded size. A training step
allocates and frees buffers of the same sizes every time. Without the cache, the system
allocator returns their pages to the OS, and the next step pays page faults to touch them
again.

## Measurements

These are medians on an Intel Core i9-9900K (8 cores, 16 hardware threads) under macOS, with Julia 1.13 and Float32, with
the best time in parentheses. The MNIST steps are one `TenferroLux.withgradient` call, which
is one `DifferentiationInterface.value_and_gradient` call. The MLP is 784→128→64→10 with
batch 128. The CNN is Conv(5×5, 1→8) → MaxPool → Conv(3×3, 8→16) → MaxPool → Dense(400→10)
with batch 128. Losses and gradients are bitwise identical in every column.

| Workload | before (44aac49) | shared engine | + derivative reuse | + 4 threads |
|:--|--:|--:|--:|--:|
| tiny eager op, `exp.(x)` with 4 elements | 16.3 µs | 12.8 µs | 12.6 µs | 15.5 µs |
| eager `reshape` (adds a graph node) | 15.2 µs | 12.4 µs | 12.0 µs | 12.7 µs |
| eager matmul, 128×784 · 784×128 | 342 µs | 321 µs | 316 µs | 189 µs |
| MLP step, `AutoTenferro` | 24.9 (20.6) ms | 21.9 (19.6) ms | 3.3 (2.8) ms | 3.4 (2.7) ms |
| MLP step, `AutoTenferroTraced` | 23.1 (18.2) ms | 18.5 (16.6) ms | 4.9 (4.2) ms | 4.4 (4.1) ms |
| CNN step, `AutoTenferro` | 246 (224) ms | 201 (189) ms | 71 (69) ms | 47 (45) ms |
| CNN step, `AutoTenferroTraced` | 273 (234) ms | 193 (188) ms | 71 (67) ms | 41 (38) ms |

For reference, Zygote takes 0.65 ms for the MLP step and 30 ms for the CNN step on the same
machine. With 8 threads, the CNN steps were slower than with 4 threads. Small models such as
the MLP are dominated by per-call overhead rather than kernel time and gain nothing from
threads.

With the native dense and cross-entropy kernels, memcpy boundary copies and the block cache,
TenferroLux's `examples/benchmark_step.jl` gives the following (median and best of two
runs, ms; BLAS and Tenferro use the same thread count; the machine had a background load of
about 3):

| Step | threads | before (9b055a4) | dense and cross-entropy kernels | Zygote |
|:--|--:|--:|--:|--:|
| MLP, `AutoTenferro` | 1 | 2.94 (2.27) | 1.23–1.29 (1.08) | 1.07–1.17 (0.88) |
| MLP, `AutoTenferroTraced` | 1 | 3.33 (2.96) | 1.43 (1.25) | |
| CNN, `AutoTenferro` | 1 | 26.9 | 19.3 (17.9) | 29.3–29.5 (27.0) |
| CNN, `AutoTenferroTraced` | 1 | 18.4 | 14.9–15.2 (13.7) | |
| MLP, `AutoTenferro` | 8 | 3.13 (2.19) | 1.39–1.46 (1.01) | 0.71–0.91 (0.49) |
| MLP, `AutoTenferroTraced` | 8 | 3.86 (2.81) | 1.42–1.59 (1.10) | |
| CNN, `AutoTenferro` | 8 | 26.2 | 18.9–19.2 (17.4) | 30.9–31.8 (28.5) |
| CNN, `AutoTenferroTraced` | 8 | 17.0 | 15.6–15.9 (14.6) | |

## Where the time goes

Each Tenferro operation crosses into Rust. A call that only adds a graph node costs about
11–12 µs, almost all of it on the Rust side: semantic recording and runtime bookkeeping. The
RustCall boundary itself costs about 1 µs. Models with many small operations are therefore
limited by the number of operations, not by kernel time.

Before each call, RustCall checks that no aggregate without a layout assertion is passed by
value. It does this at compile time only when a signature has no aggregate at all, and every
bridge method that returns a `Result` returns one: a `#[repr(C)]` result struct. Inside a
training step that runtime check cost about 8 µs per call. Tenferro decides it at compile
time for its own result structs, which carry RustCall's `FFIByValue` layout assertion; any
aggregate argument is still checked at run time. In `benchmark_step.jl` (five alternated
runs, medians of medians, ms) this took the MLP step from 1.24 to 1.11 (eager) and 1.40 to
1.22 (traced) at one thread, and from 1.49 to 1.13 and 1.61 to 1.29 at eight. Zygote took
1.21 and 0.68 in the same runs.
