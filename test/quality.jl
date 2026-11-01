using Test, Tenferro, Aqua, JET

@testset "Package quality" begin
    Aqua.test_all(Tenferro)
    ctx = EagerContext()
    x = EagerTensor(ctx, ones(2, 2))
    JET.@test_call target_modules=(Tenferro,) x + x
end
