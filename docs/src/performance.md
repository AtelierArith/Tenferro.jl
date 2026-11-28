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

## Measurements

These are medians on an Apple M-series machine with 16 cores, Julia 1.13 and Float32. The
MNIST steps are one `TenferroLux.withgradient` call, which is one
`DifferentiationInterface.value_and_gradient` call. The MLP is 784→128→64→10 with batch 128.
The CNN is Conv(5×5, 1→8) → MaxPool → Conv(3×3, 8→16) → MaxPool → Dense(400→10) with batch
128. All configurations produce bitwise identical losses and gradients.

| Workload | before (one engine per call) | shared engine, serial | 4 threads |
|:--|--:|--:|--:|
| tiny eager op, `exp.(x)` with 4 elements | 17 µs | 12.5 µs | 15 µs |
| eager matmul, 128×784 · 784×128 | 359 µs | 316 µs | 204 µs |
| MLP step, `AutoTenferro` | 38.7 ms | 23.1 ms | 24.3 ms |
| MLP step, `AutoTenferroTraced` | 19.1 ms | 21.0 ms | 19.4 ms |
| CNN step, `AutoTenferro` | 265 ms | 202 ms | 162 ms |
| CNN step, `AutoTenferroTraced` | 274 ms | 205 ms | 115 ms |
| MLP step, Zygote (reference) | 2.2 ms | | |
| CNN step, Zygote (reference) | 31 ms | | |

With 8 threads, the CNN steps took 194 ms (`AutoTenferro`) and 158 ms
(`AutoTenferroTraced`), slower than with 4 threads. Small models such as the MLP are
dominated by per-call overhead rather than kernel time and gain nothing from threads.

## Where the time goes

Each Tenferro operation crosses into Rust. A call that only adds a graph node costs about
11–12 µs, almost all of it on the Rust side: semantic recording and runtime bookkeeping. The
RustCall boundary itself costs about 1 µs. Models with many small operations are therefore
limited by the number of operations, not by kernel time.
