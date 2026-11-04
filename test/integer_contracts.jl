using Test, Tenferro, DifferentiationInterface

@testset "integer forward computation" begin
    for S in (Int32, Int64),
        (Context, TensorType) in
        ((EagerContext, EagerTensor), (TracedContext, TracedTensor))

        ctx = Context()
        a, b = S[-3, 0, 5], S[2, 3, 2]
        x, y = TensorType(ctx, a), TensorType(ctx, b)
        for op in (+, -, *, /, rem, max, min)
            actual = Array(op.(x, y))
            @test actual == op.(a, b)
            @test eltype(actual) === eltype(op.(a, b))
        end
        for op in (-, abs, sign)
            @test Array(op.(x)) == op.(a)
        end
        for op in (exp, log, sqrt, sin, cos, tanh, expm1, log1p)
            @test_throws Exception Array(op.(x))
        end
        @test_throws MethodError gradient(sum, AutoTenferro(), a)
        @test_throws MethodError gradient(sum, AutoTenferroTraced(), a)
    end
end

@testset "Boolean forward computation is explicit" begin
    for (Context, TensorType) in
        ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        ctx = Context()
        a = Bool[true, false]
        x = TensorType(ctx, a)
        @test Array(x) == a
        @test isequal(Array(x ./ x), a ./ a)
        for op in (exp, abs, sign)
            @test_throws Exception Array(op.(x))
        end
        @test_throws Exception Array(x + x)
    end
end
