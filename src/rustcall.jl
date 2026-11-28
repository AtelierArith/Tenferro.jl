# Compile-time by-value checks for the bridge's result types.
#
# Before every call, RustCall makes sure that no aggregate without a layout assertion crosses
# the boundary by value (`RustCall.ffi_check_by_value_signature`). It folds that check away
# only when no type in the signature is an aggregate. Every bridge method that returns a
# `Result` returns a `#[repr(C)]` result struct (`RustCrate.CResult_*`), so each of these
# calls ran the check at run time instead: about 8 µs per call inside a training step, or
# 150–200 µs of a 1.2 ms MNIST MLP step.
#
# The result structs are subtypes of `RustCall.FFIByValue`, RustCall's own layout assertion,
# and subtyping cannot change at run time, so for them the answer is known at compile time.
# The methods below decide it there, for this package's own result types only: arguments that
# are aggregates still get RustCall's runtime check. Without the RustCall internals this
# relies on, RustCall's default check stays in place.

# The `FFIByValue` result types RustCall generated for `RustCrate`.
function _bridge_result_types()
    isdefined(RustCall, :FFIByValue) || return DataType[]
    types = DataType[]
    for name in names(RustCrate; all=true)
        T = isdefined(RustCrate, name) ? getfield(RustCrate, name) : nothing
        T isa DataType && T <: RustCall.FFIByValue && parentmodule(T) === RustCrate && push!(types, T)
    end
    return types
end

# `nothing` when no argument type is an aggregate; otherwise RustCall's runtime check.
@generated function _check_bridge_signature(::Type{R}, ::Type{A}) where {R,A<:Tuple}
    any(RustCall.ffi_is_aggregate, A.parameters) || return :(nothing)
    return :(RustCall.ffi_check_by_value(R, $(A.parameters)))
end

if isdefined(RustCall, :ffi_check_by_value_signature) && isdefined(RustCall, :ffi_check_by_value) &&
   isdefined(RustCall, :ffi_is_aggregate)
    for T in _bridge_result_types()
        @eval RustCall.ffi_check_by_value_signature(::Type{$T}, ::Type{A}) where {A<:Tuple} =
            _check_bridge_signature($T, A)
    end
end
