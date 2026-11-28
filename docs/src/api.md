# API reference

```@meta
CurrentModule = Tenferro
```

```@docs
TypedTensor
AutoTenferro
EagerContext
EagerTensor
AutoTenferroTraced
TracedContext
TracedTensor
stopgrad
```

The NNlib, SpecialFunctions and ChainRulesCore package extensions add methods
to those packages' functions and export no names of their own; see
[Operation contracts](operations.md#Neural-network-operations).
With NNlib loaded, this includes activations, softmax, `batched_mul`, padding,
convolution (`conv`, `depthwiseconv`, `∇conv_data`, `∇conv_filter`), pooling
(`maxpool`, `meanpool`, `lpnormpool`) and upsampling (`upsample_nearest`,
`upsample_linear`/`bilinear`/`trilinear`, `pixel_shuffle`); see
[Convolution, pooling and upsampling](operations.md#Convolution,-pooling-and-upsampling).
