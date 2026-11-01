using Test, Tenferro, DifferentiationInterface

@testset "traced tensors" begin
    ctx = TracedContext()
    a = [1.0 2.0; 3.0 4.0]
    b = [2.0 4.0; 1.0 3.0]
    x, y = TracedTensor(ctx, a), TracedTensor(ctx, b)
    @test Array(x + y) ≈ a + b
    @test Array(x - y) ≈ a - b
    @test Array(x * y) ≈ a * b
    @test Array(sum(x * y))[] ≈ sum(a * b)
    @test_throws DimensionMismatch x * TracedTensor(ctx, ones(3, 1))
    @test_throws ArgumentError x + TracedTensor(TracedContext(), b)
    @test gradient(z -> sum(z * z), AutoTenferroTraced(), a) ≈ ones(2, 2) * a' + a' * ones(2, 2)
    da, db = gradient(z -> sum(z[1] * z[2]), AutoTenferroTraced(), (a, b))
    @test da ≈ ones(2, 2) * b'
    @test db ≈ a' * ones(2, 2)
    _, inactive = gradient(z -> sum(z[1]), AutoTenferroTraced(), (a, b))
    @test inactive == zeros(2, 2)
end
