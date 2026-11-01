using Test, Tenferro
using DifferentiationInterface, ADTypes, FiniteDiff, ChainRulesCore

@testset "public eager tensor and rules" begin
    ctx = EagerContext()
    x = EagerTensor(ctx, [1.0 2.0; 3.0 4.0])
    y = EagerTensor(ctx, [2.0 4.0; 1.0 3.0])
    @test Array(x + y) ≈ [3.0 6.0; 4.0 7.0]
    @test Array(x - y) ≈ [-1.0 -2.0; 2.0 1.0]
    @test Array(x * y) ≈ [4.0 10.0; 10.0 24.0]
    z, back = ChainRulesCore.rrule(*, x, y)
    seed = EagerTensor(ctx, ones(2, 2))
    _, dx, dy = back(seed)
    @test Array(dx) ≈ ones(2, 2) * Array(y)'
    @test Array(dy) ≈ Array(x)' * ones(2, 2)
    _, back_same = ChainRulesCore.rrule(*, x, x)
    _, dx1, dx2 = back_same(seed)
    @test Array(dx1) ≈ ones(2, 2) * Array(x)'
    @test Array(dx2) ≈ Array(x)' * ones(2, 2)
end

@testset "addition and subtraction pullbacks" begin
    ctx = EagerContext()
    a = EagerTensor(ctx, [1.0 2.0; 3.0 4.0])
    b = EagerTensor(ctx, [4.0 3.0; 2.0 1.0])
    seed = EagerTensor(ctx, [1.0 2.0; 3.0 4.0])
    for (op, sign) in ((+, 1.0), (-, -1.0))
        _, back = ChainRulesCore.rrule(op, a, b)
        _, da, db = back(seed)
        @test Array(da) ≈ Array(seed)
        @test Array(db) ≈ sign .* Array(seed)
        @test back(ZeroTangent()) == (NoTangent(), ZeroTangent(), ZeroTangent())
    end
    @test_throws DimensionMismatch a + EagerTensor(ctx, ones(1, 2))
    @test_throws ArgumentError a + EagerTensor(EagerContext(), ones(2, 2))
end

@testset "sum pullback" begin
    ctx = EagerContext()
    a = EagerTensor(ctx, [1.0 2.0; 3.0 4.0])
    _, back = ChainRulesCore.rrule(sum, a)
    _, da = back(2.5)
    @test Array(da) == fill(2.5, 2, 2)
    @test back(ZeroTangent()) == (NoTangent(), ZeroTangent())
end
