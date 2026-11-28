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

# Raw backend cast, used internally where the conversion is known to be exact or
# the backend behaviour is wanted: float -> int truncates toward zero, integer
# narrowing wraps, and any nonzero value becomes `true`.
function _cast(a::_ADTensor, ::Type{S}) where {S<:TensorElement}
    eltype(a) === S && return a
    id = _bridge(a, RustCrate.cast_node, RustCrate.traced_cast_node, UInt(a.id), _typed_kind(S))
    _wrap(a, S, id, a.dims)
end

# Whether every value of type `T` is exactly representable as `S`, so that the
# cast needs no value check. Casts to floating-point and complex types round to
# nearest, as Julia's `Float32(x)` does, so they count as exact here.
_exact_cast(::Type{T}, ::Type{S}) where {T,S} =
    T === S || T === Bool || (T === Int32 && S === Int64) || (S <: Complex) ||
    (S <: AbstractFloat && !(T <: Complex))

# Run Julia's own conversion `f(v)` on every element read back from `a`, so that
# a value that is not representable throws the same `InexactError` as Julia.
function _check_values(f::F, a::EagerTensor) where {F}
    foreach(f, Array(a))
    return nothing
end

function _traced_cast_error(::Type{T}, ::Type{S}) where {T,S}
    hint = if T <: Complex && !(S <: Complex)
        "use real.(x) or imag.(x)"
    elseif S === Bool
        "use x .!= 0"
    elseif T <: Integer
        "use x .% $S to wrap"
    else
        "use trunc.($S, x), round.($S, x), floor.($S, x) or ceil.($S, x)"
    end
    throw(ArgumentError("exact conversion of a TracedTensor{$T} to $S is not available: it can throw " *
                        "InexactError depending on values, which a traced graph only computes when it is " *
                        "evaluated; $hint"))
end

# `S.(x)` follows Julia's exact conversion `S(v)`: on eager tensors a value that
# is not representable in `S` (a fractional or out-of-range value cast to an
# integer, a `Bool` from anything but 0 or 1, a complex value with a nonzero
# imaginary part cast to a real type) throws `InexactError`.
_exact_convert(a::_ADTensor, ::Type{S}) where {S} =
    _exact_cast(eltype(a), S) ? _cast(a, S) : _checked_cast(a, S)
function _checked_cast(a::EagerTensor, ::Type{S}) where {S}
    _check_values(S, a)
    _cast(a, S)
end
_checked_cast(a::TracedTensor, ::Type{S}) where {S} = _traced_cast_error(eltype(a), S)

_B.broadcasted(::Type{S}, a::_ADTensor) where {S<:TensorElement} = _exact_convert(a, S)
_B.broadcasted(::Type{S}, s::EagerScalar) where {S<:TensorElement} =
    _scalar_result(_B.broadcasted(S, _scalar_tensor_value(s)))
_B.broadcasted(::typeof(float), a::_ADTensor) = _cast(a, float(eltype(a)))

# `x .% S` wraps integers to the integer type `S`, as Julia's `rem(v, S)`.
_B.broadcasted(::typeof(rem), a::Union{EagerTensor{<:Integer},TracedTensor{<:Integer}},
                ::Type{S}) where {S<:Union{Int32,Int64}} = _cast(a, S)

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

# Rounding ------------------------------------------------------------------------

const _RoundingModes = Union{typeof(RoundNearest),typeof(RoundNearestTiesAway),typeof(RoundNearestTiesUp),
                             typeof(RoundToZero),typeof(RoundFromZero),typeof(RoundUp),typeof(RoundDown)}

# `trunc` without a derivative path. Values below 2^62 in magnitude go through an
# `Int64` cast, which is exact there (larger floats are already integers) and
# carries no derivative; zeros keep the sign of `x`. Larger values and
# infinities use `x - rem(x, 1)`, which equals `x` and whose derivative `1 - 1`
# cancels (up to rounding in the accumulated cotangent).
function _trunc(x::_ADTensor)
    T = eltype(x)
    small = _B.broadcasted(<, _B.broadcasted(abs, x), T(2.0^62))
    t = _cast(_cast(_select(small, x, zero(T)), Int64), T)
    t = _select(_B.broadcasted(==, t, zero(T)), _B.broadcasted(*, x, zero(T)), t)
    large = _B.broadcasted(-, x, _B.broadcasted(rem, x, one(T)))
    large = _select(_B.broadcasted(==, _B.broadcasted(abs, x), T(Inf)), x, large)
    _select(small, t, large)
end

# Round from the truncation `t`: step one away from zero (`t + sign(x)`) where
# the mode asks for it. Every step is a function of comparisons, so the result
# carries a zero derivative.
function _round(x::_ADTensor, mode::_RoundingModes)
    T = eltype(x)
    T <: Integer && return x
    T <: Complex && throw(ArgumentError("rounding $T tensors is not supported; round real.(z) and imag.(z)"))
    t = _trunc(x)
    mode === RoundToZero && return t
    mode === RoundDown && return _select(_B.broadcasted(>, t, x), _B.broadcasted(-, t, one(T)), t)
    mode === RoundUp && return _select(_B.broadcasted(<, t, x), _B.broadcasted(+, t, one(T)), t)
    d = _B.broadcasted(abs, _B.broadcasted(-, x, t))
    away = if mode === RoundFromZero
        _B.broadcasted(>, d, zero(T))
    else
        tie = if mode === RoundNearest  # ties to even: step when `t` is odd
            _B.broadcasted(!=, _B.broadcasted(rem, _B.broadcasted(*, t, T(0.5)), one(T)), zero(T))
        elseif mode === RoundNearestTiesAway
            true
        else  # RoundNearestTiesUp
            _B.broadcasted(>, x, zero(T))
        end
        _select(_B.broadcasted(==, d, T(0.5)), tie, _B.broadcasted(>, d, T(0.5)))
    end
    _select(away, _B.broadcasted(+, t, _B.broadcasted(sign, x)), t)
end

# `round(S, x)` for an integer type `S`: round in floating point, then convert.
# On eager tensors Julia's own `f(S, v)` checks every value first, so NaN, Inf
# and out-of-range values throw its `InexactError`. Traced tensors use the
# backend cast, which rejects non-finite and out-of-range values when the graph
# is evaluated. An integer source converts exactly, as `S.(x)`.
function _round(::Type{S}, x::_ADTensor, mode::_RoundingModes, f::F) where {S,F}
    S <: Union{Int32,Int64,Bool} || throw(ArgumentError("rounding to $S is not supported; use Int32, Int64 or Bool"))
    T = eltype(x)
    T <: Integer && return _exact_convert(x, S)
    T <: Complex && throw(ArgumentError("rounding $T tensors is not supported; round real.(z) and imag.(z)"))
    if x isa EagerTensor
        _check_values(v -> f(S, v), x)
    elseif S === Bool
        throw(ArgumentError("rounding a TracedTensor to Bool is not available: it can throw InexactError " *
                            "depending on values; use x .!= 0 or compare the rounded values"))
    end
    _cast(_round(x, mode), S)
end

for (fn, mode) in ((:trunc, RoundToZero), (:floor, RoundDown), (:ceil, RoundUp), (:round, RoundNearest))
    @eval begin
        _B.broadcasted(::typeof($fn), a::_ADTensor) = _round(a, $mode)
        _B.broadcasted(::typeof($fn), ::Type{S}, a::_ADTensor) where {S<:Integer} = _round(S, a, $mode, $fn)
        _B.broadcasted(::typeof($fn), s::EagerScalar) = _scalar_result(_round(_scalar_tensor_value(s), $mode))
        _B.broadcasted(::typeof($fn), ::Type{S}, s::EagerScalar) where {S<:Integer} =
            _scalar_result(_round(S, _scalar_tensor_value(s), $mode, $fn))
    end
end
_B.broadcasted(::typeof(round), a::_ADTensor, mode::_RoundingModes) = _round(a, mode)
_B.broadcasted(::typeof(round), ::Type{S}, a::_ADTensor, mode::_RoundingModes) where {S<:Integer} =
    _round(S, a, mode, (S, v) -> round(S, v, mode))

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
