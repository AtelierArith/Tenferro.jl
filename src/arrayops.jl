# Element-type casts, public constants and contexts, and array-interface helpers
# that Lux.jl layers rely on: `vec`, mapped reductions, Boolean reductions,
# n-ary broadcasts, `repeat`, and `cat` of tensors with plain arrays.

# Contexts and constants ---------------------------------------------------------

"""
    Tenferro.context(t)

Return the context (`EagerContext` or `TracedContext`) that owns tensor `t`.
Tensors created in this context can be combined with `t`.
"""
context(t::Union{EagerTensor,EagerScalar}) = EagerContext(t.graph)
context(t::TracedTensor) = TracedContext(t.graph)

"""
    Tenferro.constant(ref, x)

Copy the array or number `x` into the context of tensor `ref` as a constant: it
is an independent input of the graph, so no derivative flows into it. A number
becomes a zero-dimensional tensor. The element type of `x` is kept; it must be
one of `Float32`, `Float64`, `ComplexF32`, `ComplexF64`, `Int32`, `Int64` or `Bool`.
"""
constant(ref::Union{_ADTensor,EagerScalar}, x::AbstractArray) = _constant_array(_reference(ref), x)
constant(ref::Union{_ADTensor,EagerScalar}, x::Number) = _constant_array(_reference(ref), fill(x))
function _constant_array(ref::_ADTensor, x::AbstractArray)
    eltype(x) <: TensorElement || throw(ArgumentError("unsupported constant element type: $(eltype(x))"))
    _constant_tensor(ref, x)
end

# Element-type casts --------------------------------------------------------------

function _cast(a::_ADTensor, ::Type{S}) where {S<:TensorElement}
    eltype(a) === S && return a
    id = _bridge(a, RustCrate.cast_node, RustCrate.traced_cast_node, UInt(a.id), _typed_kind(S))
    _wrap(a, S, id, a.dims)
end

# Julia's `Float64(z)` throws for a complex `z` with a nonzero imaginary part;
# an implicit projection would silently drop it (and its derivative).
function _check_cast(::Type{T}, ::Type{S}) where {T,S}
    T <: Complex && !(S <: Complex) &&
        throw(ArgumentError("casting $T tensors to $S would drop the imaginary part; use real.(x) or imag.(x)"))
    return nothing
end

# `S.(x)` casts to any tensor element type. Casts between floating-point and
# complex types are differentiable; casts to or from integer and `Bool` types are
# not. Floating-point values cast to integers truncate toward zero, and any
# nonzero value casts to `true`.
function _B.broadcasted(::Type{S}, a::_ADTensor) where {S<:TensorElement}
    _check_cast(eltype(a), S)
    _cast(a, S)
end
_B.broadcasted(::Type{S}, s::EagerScalar) where {S<:TensorElement} =
    _scalar_result(_B.broadcasted(S, _scalar_tensor_value(s)))
_B.broadcasted(::typeof(float), a::_ADTensor) = _cast(a, float(eltype(a)))

function _B.broadcasted(::typeof(real), a::_ADTensor)
    eltype(a) <: Complex || return a
    _cast(a, real(eltype(a)))
end
# imag(z) == real(-im * z)
function _B.broadcasted(::typeof(imag), a::_ADTensor)
    eltype(a) <: Complex || return zero(a)
    _B.broadcasted(real, _B.broadcasted(*, a, convert(eltype(a), -im)))
end

for TT in (:EagerTensor, :TracedTensor)
    @eval Base.convert(::Type{T}, a::T) where {T<:$TT} = a
    @eval function Base.convert(::Type{T}, a::$TT) where {T<:$TT}
        a isa T && return a
        S = eltype(T)
        S isa Type && S <: TensorElement || throw(ArgumentError("cannot convert $(typeof(a)) to $T"))
        b = _B.broadcasted(S, a)
        b isa T || throw(ArgumentError("cannot convert $(typeof(a)) to $T"))
        b
    end
end

# Promotions (e.g. `Float32` with `Float64`) go through the same explicit cast.
_convert_tensor(a::_ADTensor, ::Type{S}) where {S<:ADElement} = _cast(a, S)

# Arithmetic between traced scalars (eager scalars have their own methods); `+`
# and `-` are shared with tensors of any rank.
for op in (:*, :/, :^, :max, :min)
    @eval Base.$op(a::TracedTensor{<:Any,0}, b::TracedTensor{<:Any,0}) = _B.broadcasted($op, a, b)
end

# Elementwise constants -----------------------------------------------------------

_B.broadcasted(::typeof(zero), a::_ADTensor) = zero(a)
_B.broadcasted(::typeof(one), a::_ADTensor) = _constant_tensor(a, ones(eltype(a), a.dims))

Base.similar(a::_ADTensor, args...) =
    throw(ArgumentError("Tenferro tensors are immutable, so `similar` is not supported; build a new tensor " *
                        "with operations such as `zero.(x)` or `Tenferro.constant(x, array)` instead"))

# n-ary broadcasts ---------------------------------------------------------------

# Julia's `+`, `*`, `max` and `min` accept more than two arguments and fold left.
for op in (:+, :*, :max, :min)
    @eval _B.broadcasted(::_TensorStyle, ::typeof($op), a, b, c, rest...) =
        _B.broadcasted($op, _B.broadcasted($op, a, b), c, rest...)
end
# Partially applied functions with a scalar, e.g. `count(>(0), x)`.
_B.broadcasted(::_TensorStyle, f::Base.Fix2{<:Any,<:Number}, a) = _B.broadcasted(f.f, a, f.x)
_B.broadcasted(::_TensorStyle, f::Base.Fix1{<:Any,<:Number}, a) = _B.broadcasted(f.f, f.x, a)
for op in (:muladd, :fma)
    @eval _B.broadcasted(::_TensorStyle, ::typeof($op), a, b, c) = _B.broadcasted(+, _B.broadcasted(*, a, b), c)
end

# Shapes ---------------------------------------------------------------------------

Base.vec(a::_ADTensor) = ndims(a) == 1 ? a : reshape(a, length(a))

# `repeat` is a reshape and a broadcast; cotangents of repeated entries are summed.
Base.repeat(a::_ADTensor, counts::Integer...) = _repeat(a, (), counts)
function Base.repeat(a::_ADTensor; inner=nothing, outer=nothing)
    inner, outer = map(r -> r === nothing ? () : Tuple(r), (inner, outer))
    for (name, r) in (("inner", inner), ("outer", outer))
        isempty(r) || length(r) >= ndims(a) ||
            throw(ArgumentError("number of $name repetitions ($(length(r))) cannot be less than number of dimensions of input array ($(ndims(a)))"))
    end
    _repeat(a, inner, outer)
end

function _repeat(a::_ADTensor, inner::Tuple, outer::Tuple)
    all(c -> c isa Integer && c >= 1, (inner..., outer...)) ||
        throw(ArgumentError("repeat counts must be positive integers"))
    R = max(ndims(a), length(inner), length(outer))
    x = _pad_rank(a, R)
    i(d) = d <= length(inner) ? Int(inner[d]) : 1
    o(d) = d <= length(outer) ? Int(outer[d]) : 1
    all(d -> i(d) == 1 && o(d) == 1, 1:R) && return x
    # Dimension `d` becomes the triple (inner, entry, outer) in column-major order.
    expanded = ntuple(k -> (d = (k - 1) ÷ 3 + 1; mod1(k, 3) == 2 ? x.dims[d] : 1), 3R)
    target = ntuple(k -> (d = (k - 1) ÷ 3 + 1; r = mod1(k, 3); r == 1 ? i(d) : r == 2 ? x.dims[d] : o(d)), 3R)
    y = _broadcast_to(reshape(x, expanded), target)
    reshape(y, ntuple(d -> i(d) * x.dims[d] * o(d), R))
end

# Concatenation with plain arrays ---------------------------------------------------

# Plain arrays next to a tensor become constants in its context; Base's generic
# `cat` would otherwise treat the tensor as a scalar.
const _CatArg = Union{_ADTensor,AbstractArray}
_cat_tensor(ref::_ADTensor, x::_ADTensor) = x
_cat_tensor(ref::_ADTensor, x::AbstractArray) = _operand_tensor(ref, x)
function _mixed_cat(args, dims)
    ref = _reference(args...)
    tensors = map(x -> _cat_tensor(ref, x), args)
    _cat(first(tensors), Base.tail(tensors), dims)
end

Base.cat(a::_ADTensor, rest::_CatArg...; dims) = _mixed_cat((a, rest...), dims)
Base.vcat(a::_ADTensor, rest::_CatArg...) = _mixed_cat((a, rest...), 1)
Base.hcat(a::_ADTensor, rest::_CatArg...) = _mixed_cat((a, rest...), 2)
Base.cat(a::AbstractArray, b::_ADTensor, rest::_CatArg...; dims) = _mixed_cat((a, b, rest...), dims)
Base.vcat(a::AbstractArray, b::_ADTensor, rest::_CatArg...) = _mixed_cat((a, b, rest...), 1)
Base.hcat(a::AbstractArray, b::_ADTensor, rest::_CatArg...) = _mixed_cat((a, b, rest...), 2)
Base.cat(a::AbstractArray, b::AbstractArray, c::_ADTensor, rest::_CatArg...; dims) = _mixed_cat((a, b, c, rest...), dims)
Base.vcat(a::AbstractArray, b::AbstractArray, c::_ADTensor, rest::_CatArg...) = _mixed_cat((a, b, c, rest...), 1)
Base.hcat(a::AbstractArray, b::AbstractArray, c::_ADTensor, rest::_CatArg...) = _mixed_cat((a, b, c, rest...), 2)

# Mapped reductions ------------------------------------------------------------------

# `sum(f, x)` is `sum(f.(x))`; `f` must be a supported elementwise function.
for op in (:sum, :prod, :maximum, :minimum)
    @eval Base.$op(f, a::_ADTensor; kwargs...) = $op(_B.broadcasted(f, a); kwargs...)
end
Statistics.mean(f, a::_ADTensor; dims=:) = Statistics.mean(_B.broadcasted(f, a); dims=dims)

# Integer and Boolean reductions ------------------------------------------------------

const _BoolTensor = Union{EagerTensor{Bool},TracedTensor{Bool}}
const _IntegerTensor = Union{EagerTensor{<:Union{Int32,Int64,Bool}},TracedTensor{<:Union{Int32,Int64,Bool}}}

# Like Julia, sums of `Bool` values are `Int`s and means of integers are `Float64`s.
Base.sum(a::EagerTensor{Bool}; dims=:) = sum(_cast(a, Int64); dims=dims)
Base.sum(a::TracedTensor{Bool}; dims=:) = sum(_cast(a, Int64); dims=dims)
Statistics.mean(a::_IntegerTensor; dims=:) = Statistics.mean(_cast(a, Float64); dims=dims)

# `count` returns an `Int64` scalar tensor (or tensor, with `dims`).
Base.count(a::_BoolTensor; dims=:) = sum(a; dims=dims)
Base.count(f, a::_ADTensor; dims=:) = count(_bool_tensor(_B.broadcasted(f, a)); dims=dims)

_bool_tensor(a::_BoolTensor) = a
_bool_tensor(a) = throw(ArgumentError("expected Bool values, got $(eltype(a))"))

# `any`/`all` return `Bool` scalar tensors (or `Bool` tensors with `dims`); use
# `only` or `Bool(...)` to branch on an eager result.
function _any_all(a::_BoolTensor, dims, isany::Bool)
    if length(a) == 0
        value = isany ? any(Array(a); dims=dims) : all(Array(a); dims=dims)
        tensor = _constant_tensor(a, value isa AbstractArray ? value : fill(value))
        return dims isa Colon && tensor isa EagerTensor ? _scalar_result(tensor) : tensor
    end
    x = dims isa Colon ? reshape(a, length(a)) : a
    c = _reduce_dims(_cast(x, Int64), dims isa Colon ? 1 : dims, isany ? 5 : 6)
    r = _cast(c, Bool)
    dims isa Colon || return r
    r = reshape(r, ())
    r isa EagerTensor ? _scalar_result(r) : r
end
Base.any(a::_BoolTensor; dims=:) = _any_all(a, dims, true)
Base.all(a::_BoolTensor; dims=:) = _any_all(a, dims, false)
Base.any(f, a::_ADTensor; dims=:) = any(_bool_tensor(_B.broadcasted(f, a)); dims=dims)
Base.all(f, a::_ADTensor; dims=:) = all(_bool_tensor(_B.broadcasted(f, a)); dims=dims)
