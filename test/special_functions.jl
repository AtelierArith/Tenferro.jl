using Test, Tenferro, DifferentiationInterface, SpecialFunctions

@testset "error function" begin
    for T in (Float32, Float64)
        x = T[-Inf, -1, -0.0, 0.0, 0.5, Inf, NaN]
        for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            actual = Array(erf.(TensorType(Context(), x)))
            @test all(isequal.(actual[[1, 3, 4, 6, 7]], erf.(x[[1, 3, 4, 6, 7]])))
            @test actual[2:5] ≈ erf.(x[2:5])
        end
        x = T[-1, 0.5, 1]
        derivative = T(2 / sqrt(pi)) .* exp.(-x .^ 2)
        for backend in (AutoTenferro(), AutoTenferroTraced())
            @test gradient(a -> sum(erf.(a)), backend, x) ≈ derivative
        end
        direction, seed = T[0.1, -0.2, 0.3], T[-0.4, 0.5, 0.6]
        (dy,) = pushforward(a -> erf.(a), AutoTenferro(), x, (direction,))
        (back,) = pullback(a -> erf.(a), AutoTenferro(), x, (seed,))
        @test dy ≈ derivative .* direction
        @test back ≈ derivative .* seed
    end
end
