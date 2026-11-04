using Test, Tenferro, DifferentiationInterface

@testset "binary arithmetic values and directional derivatives" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = T[0.4 0.7; 1.1 1.4]
        b = T[1.3 1.6; 1.2 1.8]
        da = T[0.2 -0.1; 0.1 0.3]
        db = T[-0.1 0.2; 0.2 -0.1]
        if T <: Complex
            a .+= T(0.2im)
            b .-= T(0.1im)
            da .+= T(0.1im)
        end
        ops = T <: Complex ? (+, -, *, /, ^) : (+, -, *, /, ^, min, max)
        h = T <: Union{Float32,ComplexF32} ? 0.002f0 : 1e-6
        for op in ops
            for (Context, TensorType) in
                ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
                ctx = Context()
                @test Array(op.(TensorType(ctx, a), TensorType(ctx, b))) ≈ op.(a, b)
            end
            expected = real(sum(op.(a+h*da, b+h*db)) - sum(op.(a-h*da, b-h*db))) / (2h)
            for backend in (AutoTenferro(), AutoTenferroTraced())
                ga, gb = gradient(z -> sum(op.(z[1], z[2])), backend, (a, b))
                @test real(sum(conj.(ga) .* da) + sum(conj.(gb) .* db)) ≈ expected rtol=0.005 atol=0.002
            end
            (dy,) = pushforward(z -> op.(z[1], z[2]), AutoTenferro(), (a, b), ((da, db),))
            @test real(sum(dy)) ≈ expected rtol=0.005 atol=0.002
        end
    end
end

@testset "zero arithmetic and context checks" begin
    for (Context, TensorType) in
        ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        ctx = Context()
        a = TensorType(ctx, [0.0, 1.0, -2.0])
        b = TensorType(ctx, [2.0, 2.0, 2.0])
        @test Array(a .* b) == [0.0, 2.0, -4.0]
        @test Array(a ./ b) == [0.0, 0.5, -1.0]
        @test Array(abs.(a)) == [0.0, 1.0, 2.0]
        @test Array(sign.(a)) == [0.0, 1.0, -1.0]
        @test_throws ArgumentError a .+ TensorType(Context(), ones(3))
        @test_throws DimensionMismatch a + TensorType(ctx, ones(1, 3))
    end
end
