using Test, Tenferro, Aqua, JET, DifferentiationInterface

@testset "Package quality" begin
    Aqua.test_all(Tenferro)
    ctx = EagerContext()
    x = EagerTensor(ctx, ones(2, 2))
    JET.@test_call target_modules=(Tenferro,) x + x
    f(a) = sum(a * a)
    JET.@test_call target_modules=(Tenferro,) gradient(f, AutoTenferro(), ones(2, 2))
    elementwise(a) = sum(exp.(a .* a))
    JET.@test_call target_modules=(Tenferro,) gradient(elementwise, AutoTenferro(), ones(2, 2))
    remainder(a) = sum(rem.(a, 2.0))
    JET.@test_call target_modules=(Tenferro,) gradient(remainder, AutoTenferro(), ones(2, 2))
end
