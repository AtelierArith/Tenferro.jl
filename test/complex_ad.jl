using Test, Tenferro, DifferentiationInterface, ChainRulesCore

@testset "complex tensor autodiff" begin
    for T in (ComplexF32, ComplexF64)
        a = T[1+2im 2-im; -1+im 3+2im]
        b = T[2-im 1+2im; -3im 1-im]
        w = T[1+2im 2-im; 1-im -2+3im]
        for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            ctx = Context()
            x, y = TensorType(ctx, a), TensorType(ctx, b)
            @test Array(x) == a
            @test Array(x+y) ≈ a+b
            @test Array(x*y) ≈ a*b
        end
        for backend in (AutoTenferro(), AutoTenferroTraced())
            da, db = gradient(z -> sum(z[1] * z[2]), backend, (a, b))
            @test da isa Matrix{T}
            @test da ≈ ones(T, 2, 2) * b'
            @test db ≈ a' * ones(T, 2, 2)
            @test gradient(x -> sum(x + x), backend, T[1+im, 2-im]) == fill(T(2), 2)
        end
        ((da, db),) = pullback(z -> z[1] * z[2], AutoTenferro(), (a,b), (w,))
        @test da ≈ w * b'
        @test db ≈ a' * w
        ctx = EagerContext()
        x = EagerTensor(ctx, a)
        _, back = ChainRulesCore.rrule(sum, x)
        _, dx = back(T(2+3im))
        @test Array(dx) == fill(T(2+3im), size(a))
    end
end

@testset "tenferro built-in non-differentiable types" begin
    for a in (Int32[1 2; -3 4], Int64[1 2; -3 4], Bool[1 0; 0 1])
        t = TypedTensor(a)
        @test Array(t) == a
        t[1] = a[2]
        @test t[1] == a[2]
        for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            ctx = Context()
            x = TensorType(ctx, a)
            @test Array(x) == a
            if eltype(a) != Bool
                @test Array(x + x) == a + a
            end
        end
    end
end
