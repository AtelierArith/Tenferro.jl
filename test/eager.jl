using Test, Tenferro

@testset "public eager tensor and rules" begin
    ctx = EagerContext()
    x = EagerTensor(ctx, [1.0 2.0; 3.0 4.0])
    y = EagerTensor(ctx, [2.0 4.0; 1.0 3.0])
    @test Array(x + y) ≈ [3.0 6.0; 4.0 7.0]
    @test Array(x - y) ≈ [-1.0 -2.0; 2.0 1.0]
    @test Array(x * y) ≈ [4.0 10.0; 10.0 24.0]
    @test_throws DimensionMismatch x + EagerTensor(ctx, ones(1, 2))
    @test_throws ArgumentError x + EagerTensor(EagerContext(), ones(2, 2))
end
