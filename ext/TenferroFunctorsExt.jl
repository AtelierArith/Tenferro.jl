module TenferroFunctorsExt

using Functors
using Tenferro: EagerTensor, EagerScalar, TracedTensor

# Tensors are values: `fmap`, `fleaves` and friends must not recurse into their
# handle fields.
Functors.@leaf EagerTensor
Functors.@leaf EagerScalar
Functors.@leaf TracedTensor

end
