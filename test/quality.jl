using Test, Tenferro, Aqua, JET, DifferentiationInterface

@testset "Package quality" begin
    Aqua.test_all(Tenferro)
    ctx = EagerContext()
    x = EagerTensor(ctx, ones(2, 2))
    JET.@test_call target_modules=(Tenferro,) x + x
    f(a) = sum(a * a)
    JET.@test_call target_modules=(Tenferro,) gradient(f, AutoTenferro(), ones(2, 2))
end
