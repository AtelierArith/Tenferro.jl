using Test, Tenferro, DifferentiationInterface, Statistics

@testset "reduction shapes and values" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = reshape(T.(1:24), 2, 3, 4)
        ops = T <: Complex ? (sum, prod, mean) : (sum, prod, mean, maximum, minimum)
        for (Context, TensorType) in
            ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            x = TensorType(Context(), a)
            for op in ops, dims in (1, 2, 3, (1, 3), (3, 1), (), 4)
                y = op(x; dims)
                @test size(y) == size(op(a; dims))
                @test Array(y) ≈ op(a; dims)
            end
            @test_throws ArgumentError sum(x; dims = 0)
            @test_throws ArgumentError sum(x; dims = (1, 1))
        end
    end
end

@testset "empty reduction axes contract" begin
    for T in (Float32, Float64),
        (Context, TensorType) in
        ((EagerContext, EagerTensor), (TracedContext, TracedTensor))

        ctx = Context()
        x = TensorType(ctx, zeros(T, 0, 2))
        @test Array(sum(x; dims = 1)) == zeros(T, 1, 2)
        @test Array(prod(x; dims = 1)) == ones(T, 1, 2)
        @test_throws ArgumentError maximum(x; dims = 1)
        @test_throws ArgumentError minimum(x; dims = 1)
        @test all(isnan, Array(mean(x; dims = 1)))
        @test sum(zeros(T, 0, 2); dims = 1) == zeros(T, 1, 2)
        @test prod(zeros(T, 0, 2); dims = 1) == ones(T, 1, 2)
        @test_throws ArgumentError maximum(zeros(T, 0, 2); dims = 1)
        @test_throws ArgumentError minimum(zeros(T, 0, 2); dims = 1)
        x = TensorType(ctx, zeros(T, 2, 0))
        for op in (sum, prod, mean, maximum, minimum)
            @test size(Array(op(x; dims = 1))) == (1, 0)
        end
    end
end

@testset "zero products and tied extrema" begin
    for T in (Float32, Float64), backend in (AutoTenferro(), AutoTenferroTraced())
        a = T[0 0 2; 3 0 4]
        @test gradient(x -> sum(prod(x; dims = 1)), backend, a) ≈ T[3 0 4; 0 0 2]
        @test gradient(x -> sum(prod(x; dims = 2)), backend, a) ≈ T[0 0 0; 0 12 0]
        x = T[1 3 3; 1 2 3]
        @test gradient(a -> sum(maximum(a; dims = 1)), backend, x) ≈ T[0.5 1 0.5; 0.5 0 0.5]
        @test gradient(a -> sum(minimum(a; dims = 1)), backend, x) ≈ T[0.5 0 0.5; 0.5 1 0.5]
    end
    for T in (ComplexF32, ComplexF64), backend in (AutoTenferro(), AutoTenferroTraced())
        x = T[0, 2+im, 3-im]
        @test gradient(prod, backend, x) ≈ T[conj(prod(x[2:3])), 0, 0]
    end
end
