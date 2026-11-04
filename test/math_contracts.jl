using Test, Tenferro, DifferentiationInterface

@testset "analytic elementwise contracts" begin
    cases = (
        (exp, exp),
        (log, inv),
        (sqrt, x -> inv(2sqrt(x))),
        (sin, cos),
        (cos, x -> -sin(x)),
        (tanh, x -> 1 - tanh(x)^2),
        (expm1, exp),
        (log1p, x -> inv(1+x)),
    )
    for T in (Float32, Float64, ComplexF32, ComplexF64), (op, derivative) in cases
        x = T <: Complex ? T[0.3+0.2im, 0.7-0.1im] : T[0.3, 0.7]
        dx = T <: Complex ? T[0.2-0.1im, -0.1+0.3im] : T[0.2, -0.1]
        seed = T <: Complex ? T[0.4+0.1im, -0.2im] : T[0.4, -0.2]
        for (Context, TensorType) in
            ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            @test Array(op.(TensorType(Context(), x))) ≈ op.(x)
        end
        for backend in (AutoTenferro(), AutoTenferroTraced())
            @test gradient(a -> sum(op.(a)), backend, x) ≈ conj.(derivative.(x))
        end
        (dy,) = pushforward(a -> op.(a), AutoTenferro(), x, (dx,))
        (back,) = pullback(a -> op.(a), AutoTenferro(), x, (seed,))
        @test dy ≈ derivative.(x) .* dx
        @test back ≈ conj.(derivative.(x)) .* seed
        # Independent directional finite difference checks the derivative rule.
        h = T <: Union{Float32,ComplexF32} ? 0.002f0 : 1e-6
        @test dy ≈ (op.(x .+ h .* dx) .- op.(x .- h .* dx)) ./ (2h) rtol=0.003 atol=1e-5
    end
end

@testset "real mathematical domain boundaries" begin
    for T in (Float32, Float64),
        (Context, TensorType) in
        ((EagerContext, EagerTensor), (TracedContext, TracedTensor))

        ctx = Context()
        @test only(Array(log.(TensorType(ctx, T[0])))) == -Inf
        @test only(Array(log1p.(TensorType(ctx, T[-1])))) == -Inf
        @test only(Array(sqrt.(TensorType(ctx, T[0])))) == 0
        @test isnan(only(Array(log.(TensorType(ctx, T[-1])))))
        @test isnan(only(Array(log1p.(TensorType(ctx, T[-2])))))
        @test isnan(only(Array(sqrt.(TensorType(ctx, T[-1])))))
        @test_throws DomainError log(T(-1))
        @test_throws DomainError sqrt(T(-1))
        @test_throws DomainError log1p(T(-2))
    end
end

@testset "regular zero arguments" begin
    cases = ((exp, 1), (sin, 1), (cos, 0), (tanh, 1), (expm1, 1), (log1p, 1))
    for T in (Float32, Float64, ComplexF32, ComplexF64), (op, derivative) in cases
        x = zeros(T, 2)
        for (Context, TensorType) in
            ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            @test Array(op.(TensorType(Context(), x))) ≈ op.(x)
        end
        for backend in (AutoTenferro(), AutoTenferroTraced())
            @test gradient(a -> sum(op.(a)), backend, x) ≈ fill(T(derivative), 2)
        end
    end
end

@testset "transpose and adjoint have distinct complex pullbacks" begin
    for T in (ComplexF32, ComplexF64)
        x = T[1+im 2-im 3+2im; 4-im 5+im 6-im]
        seed = T[1+2im 3-im; 2-im 4+im; 3+im 5-2im]
        dx = x ./ 7
        for op in (transpose, adjoint)
            for (Context, TensorType) in
                ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
                @test Array(op(TensorType(Context(), x))) == op(x)
            end
            (dy,) = pushforward(op, AutoTenferro(), x, (dx,))
            (back,) = pullback(op, AutoTenferro(), x, (seed,))
            @test dy ≈ op(dx)
            @test back ≈ op(seed)
            for backend in (AutoTenferro(), AutoTenferroTraced())
                @test gradient(a -> sum(op(a) .* seed), backend, x) ≈
                      (op === transpose ? transpose(conj.(seed)) : transpose(seed))
            end
        end
    end
end
