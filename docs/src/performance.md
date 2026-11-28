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

Gradients, and each seed of a pullback, are computed for all inputs in one reverse pass. On
the traced backend, the value and all gradients are compiled and run as one program. A
`TracedContext` keeps the value of every node it has evaluated until it is closed, so
reading a value again does not recompile.

Arrays and numbers that are mixed into tensor operations, `Tenferro.constant` values and
`DifferentiationInterface.Constant` data are untracked constants. No gradient is computed
for them, and they do not take part in reverse passes.

## Measurements

These are medians on an Apple M-series machine with 16 cores, Julia 1.13 and Float32, with
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

## Where the time goes

Each Tenferro operation crosses into Rust. A call that only adds a graph node costs about
11–12 µs, almost all of it on the Rust side: semantic recording and runtime bookkeeping. The
RustCall boundary itself costs about 1 µs. Models with many small operations are therefore
limited by the number of operations, not by kernel time.
