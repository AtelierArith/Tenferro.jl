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
    params(p) = sum(p.w * p.b)
    JET.@test_call target_modules=(Tenferro,) value_and_gradient(params, AutoTenferro(), (w=ones(2, 2), b=ones(2)))
    layer(p, x) = Tenferro.logitcrossentropy(Tenferro.dense(identity, p.W, x, p.b), [1.0 0.0; 0.0 1.0])
    JET.@test_call target_modules=(Tenferro,) value_and_gradient(layer, AutoTenferro(), (W=ones(2, 3), b=ones(2)), Constant(ones(3, 2)))
    masked(a, c) = sum(ifelse.(a .> 0, a .* c, 0.0))
    JET.@test_call target_modules=(Tenferro,) gradient(masked, AutoTenferro(), ones(2), Constant(ones(2)))
end
