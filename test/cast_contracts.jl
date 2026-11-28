using Test, Tenferro
using DifferentiationInterface

const CONTEXTS = ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
const ROUND_VALUES = [-2.5, -1.5, -0.5, -0.3, -0.0, 0.0, 0.3, 0.5, 1.5, 2.5, 3.5, 0.49999999999999994,
                      -0.49999999999999994, 4.5e15 + 0.5, 2.0^62, -2.0^63, 1e300, -Inf, Inf, NaN, 3.7, -3.7]
const MODES = (RoundNearest, RoundNearestTiesAway, RoundNearestTiesUp, RoundToZero, RoundFromZero, RoundUp, RoundDown)

# The exception Julia throws for `f()` on plain values.
julia_error(f) = try
    f(); nothing
catch err
    err
end
error_text(f) = sprint(showerror, julia_error(f))

@testset "exact integer and Bool conversion on eager tensors" begin
    ctx = EagerContext()
    x = EagerTensor(ctx, [2.7, -2.7, 0.3])
    # The snippets from the issue now throw Julia's InexactError.
    @test_throws InexactError Int64.(x)
    @test error_text(() -> Int64.(x)) == error_text(() -> Int64.([2.7, -2.7, 0.3]))
    @test error_text(() -> Bool.(x)) == error_text(() -> Bool.([2.7, -2.7, 0.3]))
    big = EagerTensor(ctx, [2^40])
    @test error_text(() -> Int32.(big)) == error_text(() -> Int32.([2^40]))
    @test error_text(() -> convert(EagerTensor{Int64}, x)) == "InexactError: Int64(2.7)"

    cases = ((Int64, [1.0, -3.0, 0.0]), (Int32, [1.0f0, -3.0f0]), (Int64, [2.0^62, -2.0^63]), (Bool, [0.0, 1.0, -0.0]),
             (Bool, Int64[0, 1]), (Int32, Int64[typemax(Int32), typemin(Int32)]), (Int64, ComplexF64[3 + 0im]),
             (Float64, ComplexF64[1 + 0im]), (Bool, ComplexF64[1 + 0im]))
    for (S, v) in cases
        r = S.(EagerTensor(ctx, v))
        @test r isa EagerTensor{S,1}
        @test Array(r) == S.(v)
        @test Array(convert(EagerTensor{S}, EagerTensor(ctx, v))) == S.(v)
    end
    bad = ((Int64, [2.0^63]), (Int64, [NaN]), (Int64, [Inf]), (Int32, [3e9]), (Int32, Int64[typemax(Int32) + 1]),
           (Bool, [0.5]), (Bool, Int64[2]), (Bool, Int32[-1]), (Int64, [0.5f0]), (Float64, ComplexF64[1 + 2im]),
           (Float32, ComplexF32[0 + 1im]), (Int64, ComplexF64[2 + 1im]), (Int64, ComplexF64[2.5 + 0im]))
    for (S, v) in bad
        @test_throws InexactError S.(EagerTensor(ctx, v))
        @test error_text(() -> S.(EagerTensor(ctx, v))) == error_text(() -> S.(v))
    end
    # Always-exact casts need no check.
    @test Array(Int64.(EagerTensor(ctx, Int32[typemax(Int32), -1]))) == Int64[typemax(Int32), -1]
    @test Array(Int32.(EagerTensor(ctx, [true, false]))) == Int32[1, 0]
    @test Array(Float32.(EagerTensor(ctx, [1.0e-50, 0.1]))) == Float32.([1.0e-50, 0.1])
    @test Array(Float64.(EagerTensor(ctx, Int64[2^60 + 1]))) == Float64.([2^60 + 1])
    # Empty and zero-dimensional tensors.
    @test size(Int64.(EagerTensor(ctx, Float64[]))) == (0,)
    s = sum(EagerTensor(ctx, [1.0, 2.0]))
    @test Int64.(s)[] == 3
    @test_throws InexactError Int64.(sum(EagerTensor(ctx, [1.0, 0.5])))
    @test Int64(s) === 3
    @test_throws InexactError Int64(sum(EagerTensor(ctx, [1.0, 0.5])))
end

@testset "exact conversion is rejected on traced tensors" begin
    ctx = TracedContext()
    x = TracedTensor(ctx, [2.7, -2.7, 0.3])
    for S in (Int32, Int64, Bool)
        err = julia_error(() -> S.(x))
        @test err isa ArgumentError
        @test occursin("TracedTensor", err.msg)
        @test occursin(S === Bool ? ".!= 0" : "trunc.($S, x)", err.msg)
        @test_throws ArgumentError convert(TracedTensor{S}, x)
    end
    @test occursin("x .% Int32", julia_error(() -> Int32.(TracedTensor(ctx, [1, 2]))).msg)
    @test_throws ArgumentError Bool.(TracedTensor(ctx, Int64[0, 1]))
    @test occursin("real.(x)", julia_error(() -> Float64.(TracedTensor(ctx, ComplexF64[1 + 0im]))).msg)
    # Always-exact casts still work.
    @test Array(Int64.(TracedTensor(ctx, Int32[1, -2]))) == [1, -2]
    @test Array(Int64.(TracedTensor(ctx, [true, false]))) == [1, 0]
    @test Array(Float32.(x)) == Float32.([2.7, -2.7, 0.3])
    @test Array(ComplexF64.(x)) == ComplexF64.([2.7, -2.7, 0.3])
end

@testset "wrapping integer conversion with .%" begin
    v = Int64[2^40, 2^31, -1, 5]
    for (Context, TensorType) in CONTEXTS
        ctx = Context()
        r = TensorType(ctx, v) .% Int32
        @test r isa TensorType{Int32,1}
        @test Array(r) == v .% Int32
        @test Array(TensorType(ctx, [true, false]) .% Int64) == [1, 0]
    end
end

@testset "float-valued rounding matches Base" begin
    for (Context, TensorType) in CONTEXTS, T in (Float64, Float32)
        v = T.(ROUND_VALUES)
        x = TensorType(Context(), v)
        for (f, expected) in ((trunc, trunc.(v)), (floor, floor.(v)), (ceil, ceil.(v)), (round, round.(v)))
            r = f.(x)
            @test r isa TensorType{T,1}
            @test isequal(Array(r), expected)  # also checks NaN and the sign of zero
        end
        for mode in MODES
            @test isequal(Array(round.(x, mode)), round.(v, mode))
        end
        # Integer tensors round to themselves.
        @test Array(round.(TensorType(Context(), Int32[3, -4]))) == Int32[3, -4]
    end
    # Ties go to even.
    x = EagerTensor(EagerContext(), [0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    @test Array(round.(x)) == [0.0, 2.0, 2.0, -0.0, -2.0, -2.0]
    s = sum(EagerTensor(EagerContext(), [1.25, 1.25]))
    @test round.(s)[] == 2.0
    @test floor.(Int64, s)[] == 2
    @test_throws ArgumentError round.(EagerTensor(EagerContext(), ComplexF64[1.5 + 0im]))
end

@testset "integer-valued rounding" begin
    v = [-2.5, -1.5, -0.7, -0.5, 0.0, 0.3, 0.5, 1.5, 2.5, 2.7, 1e9]
    for (Context, TensorType) in CONTEXTS, S in (Int32, Int64), T in (Float64, Float32)
        x = TensorType(Context(), T.(v))
        for f in (trunc, floor, ceil, round)
            r = f.(S, x)
            @test r isa TensorType{S,1}
            @test Array(r) == f.(S, T.(v))
        end
        for mode in MODES
            @test Array(round.(S, x, mode)) == round.(S, T.(v), mode)
        end
    end
    ctx = EagerContext()
    for v in ([NaN], [Inf], [-Inf], [3e9], [2.0^63], [-2.0^63 - 2^11])
        for S in (Int32, Int64), f in (trunc, floor, ceil, round)
            if julia_error(() -> f.(S, v)) === nothing
                @test Array(f.(S, EagerTensor(ctx, v))) == f.(S, v)
            else
                @test_throws InexactError f.(S, EagerTensor(ctx, v))
                @test error_text(() -> f.(S, EagerTensor(ctx, v))) == error_text(() -> f.(S, v))
            end
        end
    end
    @test error_text(() -> round.(Int64, EagerTensor(ctx, [NaN]), RoundNearestTiesAway)) ==
          error_text(() -> round.(Int64, [NaN], RoundNearestTiesAway))
    # Bool targets follow Julia on eager tensors and are rejected on traced ones.
    @test Array(round.(Bool, EagerTensor(ctx, [0.3, 0.7]))) == [false, true]
    @test_throws InexactError round.(Bool, EagerTensor(ctx, [1.7]))
    @test_throws ArgumentError round.(Bool, TracedTensor(TracedContext(), [0.3]))
    # Integer sources convert exactly.
    @test Array(round.(Int32, EagerTensor(ctx, Int64[3, -4]))) == Int32[3, -4]
    @test_throws InexactError round.(Int32, EagerTensor(ctx, [2^40]))
    @test_throws ArgumentError round.(Int32, TracedTensor(TracedContext(), [2^40]))
    @test Array(trunc.(Int64, TracedTensor(TracedContext(), Int32[7]))) == [7]
    # Traced: the backend cast rejects non-finite and out-of-range values when
    # the graph is evaluated.
    @test_throws Exception Array(trunc.(Int64, TracedTensor(TracedContext(), [NaN])))
    @test_throws Exception Array(round.(Int32, TracedTensor(TracedContext(), [3e9])))
end

@testset "rounding carries no derivative" begin
    x0 = [1.3, -2.7, 0.5, 2.5, -0.2, 1e300]
    for backend in (AutoTenferro(), AutoTenferroTraced())
        for f in (trunc, floor, ceil, round)
            @test gradient(x -> sum(f.(x)), backend, x0) == zeros(6)
            # d/dx (f(x) * x) == f(x), with no rounding error from the zero derivative
            @test gradient(x -> sum(f.(x) .* x), backend, x0) == f.(x0)
        end
        @test gradient(x -> sum(round.(x, RoundNearestTiesAway) .* x), backend, x0) == round.(x0, RoundNearestTiesAway)
        @test gradient(x -> sum(Float64.(floor.(Int64, x[1:5])) .* x[1:5]), backend, x0) == [floor.(x0[1:5]); 0.0]
        @test gradient(x -> sum(Float32.(x) .* 2), backend, x0) == fill(2.0, 6)
        @test only(pushforward(x -> floor.(x) .* x, backend, x0, (ones(6),))) == floor.(x0)
        @test only(pushforward(x -> round.(x), backend, x0, (ones(6),))) == zeros(6)
    end
end
