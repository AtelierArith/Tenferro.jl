# Tenferro.jl

Tenferro.jl is an experimental Julia interface to [tenferro-rs](https://github.com/tensor4all/tenferro-rs). It uses RustCall.jl for Rust interoperability and provides an automatic differentiation backend for DifferentiationInterface.jl.

Start with [Getting started](getting-started.md), then see [Automatic differentiation](autodiff.md) for gradients and pullbacks.

The current implementation supports CPU arrays with real and complex floating-point elements, matrix multiplication, addition, subtraction, and `sum`. The API and supported operations are expected to grow.
