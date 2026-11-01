using Test, Tenferro
using DifferentiationInterface, ADTypes, FiniteDiff, ChainRulesCore

@testset "AutoTenferro matrix gradient" begin
    x = [1.0 2.0; 3.0 4.0]
    f(a) = sum(a * a)
    g = DifferentiationInterface.gradient(f, AutoTenferro(), x)
    @test g ≈ ones(2, 2) * transpose(x) + transpose(x) * ones(2, 2)
    @test DifferentiationInterface.gradient(f, AutoTenferro(), zeros(1, 1)) ≈ zeros(1, 1)
    @test_throws DimensionMismatch DifferentiationInterface.gradient(f, AutoTenferro(), ones(2, 3))
end

@testset "AutoTenferro composes actual operations" begin
    x = [1.0 -0.4; 0.7 2.0]
    f(a) = sum(a * a * a)
    got = DifferentiationInterface.gradient(f, AutoTenferro(), x)
    reference = DifferentiationInterface.gradient(f, AutoFiniteDiff(), x)
    @test got ≈ reference rtol=1e-6
    @test_throws ArgumentError DifferentiationInterface.gradient(identity, AutoTenferro(), x)
end

@testset "AutoTenferro multiple matrix inputs" begin
    a = [1.0 2.0 3.0; 4.0 5.0 6.0]
    b = [2.0 1.0; -1.0 3.0; 4.0 -2.0]
    w = [1.0 -3.0; 2.0 0.5]
    f(z) = z[1] * z[2]
    ((da, db),) = DifferentiationInterface.pullback(f, AutoTenferro(), (a, b), (w,))
    @test da ≈ w * b'
    @test db ≈ a' * w
    (da, db) = DifferentiationInterface.gradient(z -> sum(f(z)), AutoTenferro(), (a, b))
    @test da ≈ ones(2, 2) * b'
    @test db ≈ a' * ones(2, 2)
    @test_throws DimensionMismatch DifferentiationInterface.pullback(f, AutoTenferro(), (a, b), (ones(1, 1),))
end

@testset "AutoTenferro inactive inputs and multiple seeds" begin
    a, b = [1.0 2.0; 3.0 4.0], [2.0 0.0; 0.0 2.0]
    seeds = (ones(2, 2), [1.0 2.0; 3.0 4.0])
    outputs = DifferentiationInterface.pullback(z -> z[1] * z[1], AutoTenferro(), (a, b), seeds)
    @test length(outputs) == 2
    @test outputs[1][2] == zeros(2, 2)
    @test outputs[2][1] ≈ seeds[2] * a' + a' * seeds[2]
    da, db = DifferentiationInterface.gradient(z -> sum(z[1] * z[1]), AutoTenferro(), (a, b))
    @test da ≈ ones(2, 2) * a' + a' * ones(2, 2)
    @test db == zeros(2, 2)
end


const rule_calls = Ref(0)
probe(x::EagerTensor) = Tenferro._apply_rule(probe, (x,), () -> x + x)
function ChainRulesCore.rrule(::typeof(probe), x::EagerTensor)
    rule_calls[] += 1
    y = x + x
    back(seed) = (NoTangent(), seed + seed)
    return y, back
end

@testset "AutoTenferro invokes registered rules" begin
    rule_calls[] = 0
    x = [1.0 2.0; 3.0 4.0]
    @test DifferentiationInterface.gradient(a -> sum(probe(a)), AutoTenferro(), x) ≈ fill(2.0, 2, 2)
    @test rule_calls[] == 1
end
