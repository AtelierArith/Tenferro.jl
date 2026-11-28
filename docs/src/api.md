# API reference

```@meta
CurrentModule = Tenferro
```

```@docs
TypedTensor
AutoTenferro
EagerContext
EagerTensor
EagerScalar
AutoTenferroTraced
TracedContext
TracedTensor
stopgrad
constant
context
DifferentiableElement
num_threads
set_num_threads
```

Element-type conversion and rounding use Base's functions on tensors: `S.(x)`,
`convert(EagerTensor{S}, x)`, `float.`, `real.`, `imag.`, `trunc.`, `floor.`,
`ceil.`, `round.` (also `round.(S, x, mode)`) and `x .% S`. Value-dependent
conversions throw `InexactError` on eager tensors and `ArgumentError` on traced
tensors; see [Element-type casts and constants](operations.md#Element-type-casts-and-constants).

The NNlib, SpecialFunctions and ChainRulesCore package extensions add methods
to those packages' functions and export no names of their own; see
[Operation contracts](operations.md#Neural-network-operations).
With NNlib loaded, this includes activations, softmax, `batched_mul`, padding,
convolution (`conv`, `depthwiseconv`, `∇conv_data`, `∇conv_filter`), pooling
(`maxpool`, `meanpool`, `lpnormpool`) and upsampling (`upsample_nearest`,
`upsample_linear`/`bilinear`/`trilinear`, `pixel_shuffle`); see
[Convolution, pooling and upsampling](operations.md#Convolution,-pooling-and-upsampling),
and `NNlib.gather` and `NNlib.scatter` (`+`, `-`, `mean`).
The Functors extension (`TenferroFunctorsExt`) makes `EagerTensor`,
`TracedTensor` and eager scalars Functors.jl leaves, so `fmap` and `fleaves`
treat each tensor as one value.
