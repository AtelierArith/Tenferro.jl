# Basic array-interface operations built on the tenferro-rs bridge.

_wrap(a::_ADTensor, ::Type{T}, id, dims) where {T} = _factor_tensor(a, T, id, dims)
_bridge(a::_ADTensor, eager, traced, args...) =
    RustCall.result_to_exception(_rust_call(a.graph, a isa EagerTensor ? eager : traced, args...))
_same_graph(a::_ADTensor, b::_ADTensor) =
    a.graph === b.graph || throw(ArgumentError("tensors belong to different contexts"))

# Scalar values ---------------------------------------------------------------

const _ScalarTensor = Union{EagerScalar,TracedTensor{<:Any,0}}

Base.Array(s::EagerScalar) = Array(_scalar_tensor_value(s))
Base.getindex(s::EagerScalar) = Array(s)[]
Base.only(s::_ScalarTensor) = Array(s)[]
for T in (Float32, Float64, ComplexF32, ComplexF64, Int32, Int64, Bool)
    @eval begin
        Base.convert(::Type{$T}, s::_ScalarTensor) = convert($T, only(s))
        (::Type{$T})(s::_ScalarTensor) = convert($T, only(s))
    end
end
Base.size(::EagerScalar) = ()
Base.ndims(::EagerScalar) = 0
Base.length(::EagerScalar) = 1

# Comparisons and selection ---------------------------------------------------

const _Operand = Union{_ADTensor,EagerScalar,Number,AbstractArray}

_operand_tensor(::_ADTensor, x::_ADTensor) = x
_operand_tensor(::_ADTensor, x::EagerScalar) = _scalar_tensor_value(x)
function _operand_tensor(ref::_ADTensor, x::Number)
    S = promote_type(eltype(ref), typeof(x))
    S <: TensorElement || throw(ArgumentError("unsupported promoted tensor element type: $S"))
    _constant_tensor(ref, fill(convert(S, x)))
end
function _operand_tensor(ref::_ADTensor, x::AbstractArray)
    eltype(x) <: TensorElement || throw(ArgumentError("unsupported array element type: $(eltype(x))"))
    _constant_tensor(ref, x)
end

function _compare(a::_ADTensor, b::_ADTensor, code)
    x, y = _broadcast_pair(a, b)
    id = _bridge(x, RustCrate.compare_node, RustCrate.traced_compare_node, UInt(x.id), UInt(y.id), Int32(code))
    _wrap(x, Bool, id, x.dims)
end

for (op, code) in ((:(==), 0), (:<, 1), (:<=, 2), (:>, 3), (:>=, 4))
    @eval begin
        Base.Broadcast.broadcasted(::typeof($op), a::_ADTensor, b::_ADTensor) = _compare(a, b, $code)
        Base.Broadcast.broadcasted(::typeof($op), a::_ADTensor, b::Union{EagerScalar,Number,AbstractArray}) =
            _compare(a, _operand_tensor(a, b), $code)
        Base.Broadcast.broadcasted(::typeof($op), a::Union{EagerScalar,Number,AbstractArray}, b::_ADTensor) =
            _compare(_operand_tensor(b, a), b, $code)
    end
end
Base.Broadcast.broadcasted(::typeof(!=), a::_Operand, b::_ADTensor) = _not(Base.Broadcast.broadcasted(==, a, b))
Base.Broadcast.broadcasted(::typeof(!=), a::_ADTensor, b::Union{EagerScalar,Number,AbstractArray}) =
    _not(Base.Broadcast.broadcasted(==, a, b))
_not(mask::_ADTensor) = _select(mask, false, true)
Base.Broadcast.broadcasted(::typeof(!), mask::Union{EagerTensor{Bool},TracedTensor{Bool}}) = _not(mask)

_reference(x::_ADTensor, rest...) = x
_reference(x::EagerScalar, rest...) = _scalar_tensor_value(x)
_reference(_, rest...) = _reference(rest...)
_reference() = throw(ArgumentError("at least one Tenferro tensor is required"))

_operand_eltype(x::Union{_ADTensor,EagerScalar}) = eltype(x)
_operand_eltype(x::Number) = typeof(x)
_operand_eltype(x::AbstractArray) = eltype(x)
_operand_size(x::Union{_ADTensor,EagerScalar}) = size(x)
_operand_size(x::Number) = ()
_operand_size(x::AbstractArray) = size(x)

_lift_tensor(ref::_ADTensor, x::_ADTensor, ::Type) = (_same_graph(ref, x); x)
_lift_tensor(::_ADTensor, x::EagerScalar, ::Type) = _scalar_tensor_value(x)
_lift_tensor(ref::_ADTensor, x::Number, ::Type{T}) where {T} = _constant_tensor(ref, fill(convert(T, x)))
_lift_tensor(ref::_ADTensor, x::AbstractArray, ::Type{T}) where {T} = _constant_tensor(ref, convert(AbstractArray{T}, x))

function _lift(ref::_ADTensor, x, ::Type{T}, dims) where {T}
    t = _lift_tensor(ref, x, T)
    if eltype(t) !== T
        T <: ADElement || throw(ArgumentError("cannot convert $(eltype(t)) operand to $T"))
        t = _convert_tensor(t, T)
    end
    _broadcast_to(t, dims)
end

function _select(condition, on_true, on_false)
    ref = _reference(condition, on_true, on_false)
    _operand_eltype(condition) === Bool || throw(ArgumentError("ifelse condition must contain Bool values"))
    T = promote_type(_operand_eltype(on_true), _operand_eltype(on_false))
    T <: TensorElement || throw(ArgumentError("unsupported promoted tensor element type: $T"))
    dims = Base.Broadcast.broadcast_shape(_operand_size(condition), _operand_size(on_true), _operand_size(on_false))
    c = _lift(ref, condition, Bool, dims)
    t = _lift(ref, on_true, T, dims)
    f = _lift(ref, on_false, T, dims)
    id = _bridge(ref, RustCrate.select_node, RustCrate.traced_select_node, UInt(c.id), UInt(t.id), UInt(f.id))
    _wrap(ref, T, id, dims)
end

const _Mask = Union{Bool,AbstractArray{Bool}}
Base.Broadcast.broadcasted(::typeof(ifelse), c::Union{_ADTensor,EagerScalar}, a::_Operand, b::_Operand) = _select(c, a, b)
Base.Broadcast.broadcasted(::typeof(ifelse), c::_Mask, a::Union{_ADTensor,EagerScalar}, b::_Operand) = _select(c, a, b)
Base.Broadcast.broadcasted(::typeof(ifelse), c::_Mask, a::Union{Number,AbstractArray}, b::Union{_ADTensor,EagerScalar}) = _select(c, a, b)

# Broadcasting between tensors and eager scalars.
for op in (:+, :-, :*, :/, :^, :rem, :max, :min)
    @eval begin
        Base.Broadcast.broadcasted(::typeof($op), a::EagerTensor, b::EagerScalar) =
            Base.Broadcast.broadcasted($op, a, _scalar_tensor_value(b))
        Base.Broadcast.broadcasted(::typeof($op), a::EagerScalar, b::EagerTensor) =
            Base.Broadcast.broadcasted($op, _scalar_tensor_value(a), b)
        Base.Broadcast.broadcasted(::typeof($op), a::EagerScalar, b::AbstractArray) =
            Base.Broadcast.broadcasted($op, _scalar_tensor_value(a), b)
        Base.Broadcast.broadcasted(::typeof($op), a::AbstractArray, b::EagerScalar) =
            Base.Broadcast.broadcasted($op, a, _scalar_tensor_value(b))
        Base.Broadcast.broadcasted(::typeof($op), a::EagerScalar, b::Union{EagerScalar,Number}) =
            _scalar_result(Base.Broadcast.broadcasted($op, _scalar_tensor_value(a), _operand_tensor(_scalar_tensor_value(a), b)))
        Base.Broadcast.broadcasted(::typeof($op), a::Number, b::EagerScalar) =
            _scalar_result(Base.Broadcast.broadcasted($op, _operand_tensor(_scalar_tensor_value(b), a), _scalar_tensor_value(b)))
    end
end

# Elementwise helpers ---------------------------------------------------------

function Base.Broadcast.broadcasted(::typeof(abs2), a::_ADTensor)
    eltype(a) <: Complex || return Base.Broadcast.broadcasted(*, a, a)
    Base.Broadcast.broadcasted(^, Base.Broadcast.broadcasted(abs, a), 2)
end
Base.Broadcast.broadcasted(::typeof(inv), a::_ADTensor) = Base.Broadcast.broadcasted(/, one(eltype(a)), a)
Base.zero(a::_ADTensor) = _constant_tensor(a, zeros(eltype(a), a.dims))
Base.zero(s::EagerScalar) = _scalar_result(zero(_scalar_tensor_value(s)))
Base.copy(a::_ADTensor) = a
Base.copy(s::EagerScalar) = s

# Shape helpers ---------------------------------------------------------------

function Base.dropdims(a::_ADTensor; dims)
    axes = dims isa Integer ? (dims,) : Tuple(dims)
    all(d -> d isa Integer && 1 <= d <= ndims(a), axes) || throw(ArgumentError("dropped dims must be in range 1:ndims(A)"))
    length(unique(axes)) == length(axes) || throw(ArgumentError("dropped dims must be unique"))
    all(d -> a.dims[d] == 1, axes) || throw(ArgumentError("dropped dims must all be size 1"))
    reshape(a, Tuple(a.dims[i] for i in eachindex(a.dims) if !(i in axes)))
end

function Base.selectdim(a::_ADTensor, d::Integer, i)
    1 <= d <= ndims(a) || throw(ArgumentError("dimension $d out of range"))
    a[ntuple(k -> k == d ? i : Colon(), ndims(a))...]
end

function Base.eachslice(a::_ADTensor; dims, drop::Bool=true)
    axes = dims isa Integer ? (dims,) : Tuple(dims)
    all(d -> d isa Integer && 1 <= d <= ndims(a), axes) || throw(ArgumentError("slice dims must be in range 1:ndims(A)"))
    length(unique(axes)) == length(axes) || throw(ArgumentError("slice dims must be unique"))
    slices = map(CartesianIndices(map(d -> a.dims[d], axes))) do I
        a[ntuple(k -> (j = findfirst(==(k), axes)) === nothing ? Colon() : I[j], ndims(a))...]
    end
    drop && return slices
    reshape(slices, ntuple(k -> k in axes ? a.dims[k] : 1, ndims(a)))
end

function _pad_rank(a::_ADTensor, rank)
    ndims(a) == rank && return a
    reshape(a, ntuple(i -> i <= ndims(a) ? a.dims[i] : 1, rank))
end
Base.vcat(a::_ADTensor, rest::_ADTensor...) = cat(a, rest...; dims=1)
Base.hcat(a::_ADTensor, rest::_ADTensor...) = cat(a, rest...; dims=2)

# Indexing --------------------------------------------------------------------

const _IndexArg = Union{Integer,AbstractVector{<:Integer},Colon}

function _gather_axis(a::_ADTensor, indices::AbstractVector{<:Integer}, axis)
    ix = _constant_tensor(a, reshape(Int64.(indices) .- 1, :, 1))
    id = _bridge(a, RustCrate.gather_axis, RustCrate.traced_gather_axis, UInt(a.id), UInt(ix.id), UInt(axis - 1))
    _wrap(a, eltype(a), id, ntuple(i -> i == axis ? length(indices) : a.dims[i], ndims(a)))
end

_slice_range(i::Colon, n) = 1:n
_slice_range(i::Integer, n) = i:i
_slice_range(i::AbstractRange{<:Integer}, n) = i
_slice_range(i::AbstractVector{<:Integer}, n) = 1:n

function _index(a::_ADTensor, indices::Tuple)
    if length(indices) == 1 && only(indices) isa Colon
        return ndims(a) == 1 ? a : reshape(a, length(a))
    end
    length(indices) == ndims(a) || throw(ArgumentError("one index per dimension is required"))
    original = indices
    indices = map(original, a.dims) do i, n
        i isa AbstractVector{Bool} || return i
        length(i) == n || throw(BoundsError(a, original))
        findall(i)
    end
    ranges = map(_slice_range, indices, a.dims)
    for (i, r, n) in zip(indices, ranges, a.dims)
        if i isa AbstractRange || !(i isa AbstractVector)
            isempty(r) && throw(ArgumentError("empty slices are not supported"))
            step(r) > 0 || throw(ArgumentError("slice steps must be positive"))
            first(r) >= 1 && last(r) <= n || throw(BoundsError(a, indices))
        else
            isempty(i) && throw(ArgumentError("empty index vectors are not supported"))
            all(j -> 1 <= j <= n, i) || throw(BoundsError(a, indices))
        end
    end
    result = a
    if any(((r, n),) -> r != 1:n, zip(ranges, a.dims))
        values = UInt[map(r -> first(r) - 1, ranges)..., map(last, ranges)..., map(step, ranges)...]
        result = _slice(a, values, map(length, ranges))
    end
    for (axis, i) in enumerate(indices)
        if i isa AbstractVector && !(i isa AbstractRange)
            result = _gather_axis(result, i, axis)
        end
    end
    out_dims = Tuple(result.dims[d] for d in eachindex(indices) if !(indices[d] isa Integer))
    output = out_dims == result.dims ? result : reshape(result, out_dims)
    isempty(out_dims) && output isa EagerTensor && return _scalar_result(output)
    return output
end
Base.getindex(a::_ADTensor, indices::_IndexArg...) = _index(a, indices)

# Statistics -------------------------------------------------------------------

_sqrt(x::Union{EagerScalar,TracedTensor{<:Any,0}}) = sqrt(x)
_sqrt(x::_ADTensor) = Base.Broadcast.broadcasted(sqrt, x)

function Statistics.var(a::_ADTensor; corrected::Bool=true, mean=nothing, dims=:)
    m = mean === nothing ? Statistics.mean(a; dims=dims) : mean
    centered = Base.Broadcast.broadcasted(-, a, m)
    n = dims isa Colon ? length(a) : prod(d -> size(a, d), dims isa Integer ? (dims,) : Tuple(dims); init=1)
    sum(Base.Broadcast.broadcasted(abs2, centered); dims=dims) / (n - Int(corrected))
end
Statistics.std(a::_ADTensor; corrected::Bool=true, mean=nothing, dims=:) =
    _sqrt(Statistics.var(a; corrected=corrected, mean=mean, dims=dims))

# Stop-gradient -----------------------------------------------------------------

"""
    Tenferro.stopgrad(x)

Return the value of `x` with gradient flow blocked: derivatives computed by
`AutoTenferro` or `AutoTenferroTraced` treat the result as a constant.
Eager tensors use the backend's `detach`; traced tensors are evaluated and
re-inserted into their graph as constant inputs. Tuples and `NamedTuple`s are
processed elementwise, and other values are returned unchanged.
"""
function stopgrad(x::EagerTensor{T,N}) where {T,N}
    id = RustCall.result_to_exception(_rust_call(x.graph, RustCrate.detach_node, UInt(x.id)))
    EagerTensor{T,N,typeof(x.graph)}(x.graph, id, x.dims)
end
stopgrad(x::EagerScalar) = _scalar_result(stopgrad(_scalar_tensor_value(x)))
stopgrad(x::TracedTensor) = _constant_tensor(x, Array(x))
stopgrad(x::Union{Tuple,NamedTuple}) = map(stopgrad, x)
stopgrad(x) = x

# Fused broadcasts of plain Julia arrays stay lazy; materialize them before they
# meet a tensor operand.
const _Lazy = Base.Broadcast.Broadcasted
for op in (:+, :-, :*, :/, :^, :rem, :max, :min, :(==), :<, :<=, :>, :>=, :!=)
    @eval begin
        Base.Broadcast.broadcasted(::typeof($op), a::Union{_ADTensor,EagerScalar}, b::_Lazy) =
            Base.Broadcast.broadcasted($op, a, Base.Broadcast.materialize(b))
        Base.Broadcast.broadcasted(::typeof($op), a::_Lazy, b::Union{_ADTensor,EagerScalar}) =
            Base.Broadcast.broadcasted($op, Base.Broadcast.materialize(a), b)
    end
end
Base.Broadcast.broadcasted(::typeof(ifelse), c::_Lazy, a::Union{_ADTensor,EagerScalar}, b::Union{_Operand,_Lazy}) =
    _select(Base.Broadcast.materialize(c), a, _eager_operand(b))
Base.Broadcast.broadcasted(::typeof(ifelse), c::_Lazy, a::Union{Number,AbstractArray,_Lazy}, b::Union{_ADTensor,EagerScalar}) =
    _select(Base.Broadcast.materialize(c), _eager_operand(a), b)
Base.Broadcast.broadcasted(::typeof(ifelse), c::Union{_ADTensor,EagerScalar}, a::_Lazy, b::Union{_Operand,_Lazy}) =
    _select(c, Base.Broadcast.materialize(a), _eager_operand(b))
Base.Broadcast.broadcasted(::typeof(ifelse), c::_Mask, a::_Lazy, b::Union{_ADTensor,EagerScalar}) =
    _select(c, Base.Broadcast.materialize(a), b)
Base.Broadcast.broadcasted(::typeof(ifelse), c::Union{_ADTensor,EagerScalar}, a::_Operand, b::_Lazy) =
    _select(c, a, Base.Broadcast.materialize(b))
_eager_operand(x::_Lazy) = Base.Broadcast.materialize(x)
_eager_operand(x) = x
