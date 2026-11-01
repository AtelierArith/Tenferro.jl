using Test

@testset "extension activation" begin
    project = dirname(Base.active_project())
    code = "using Tenferro; @assert isnothing(Base.get_extension(Tenferro, :TenferroChainRulesCoreExt)); using ChainRulesCore; @assert !isnothing(Base.get_extension(Tenferro, :TenferroChainRulesCoreExt))"
    @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project -e $code`)
end

using Test, Tenferro
using ChainRulesCore

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

@testset "pullback graph reuse and aliasing" begin
    ctx = EagerContext()
    a = ComplexF64[1+2im 3-im; 2im 1+im]
    x = EagerTensor(ctx, a)
    z, back = rrule(*, x, x)
    cot = EagerTensor(ctx, ComplexF64[1+im 3im; -2 2-im])
    _, dx, dy = back(cot)
    @test Array(dx) ≈ Array(cot) * a'
    @test Array(dy) ≈ a' * Array(cot)
    @test Array(dx + dy) ≈ Array(cot) * a' + a' * Array(cot)
    _, dx2, dy2 = back(cot)
    @test Array(dx2) ≈ Array(dx)
    @test Array(dy2) ≈ Array(dy)
    other_seed = EagerTensor(EagerContext(), Array(cot))
    _, dx3, dy3 = back(other_seed)
    @test Array(dx3) ≈ Array(dx)
    @test Array(dy3) ≈ Array(dy)
end


@testset "complex sum cotangent" begin
    for T in (ComplexF32, ComplexF64)
        x = EagerTensor(EagerContext(), T[1+im, 2-im])
        _, back = rrule(sum, x)
        _, dx = back(T(2+3im))
        @test Array(dx) == fill(T(2+3im), 2)
    end
end
