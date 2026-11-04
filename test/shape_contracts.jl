using Test, Tenferro, DifferentiationInterface

@testset "shape results own their Julia buffers" begin
    for (Context, TensorType) in
        ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        input = reshape(collect(1.0:12), 3, 4)
        original = copy(input)
        x = TensorType(Context(), input)
        y = permutedims(x, (2, 1))[1:2:3, :]
        input .= -100
        GC.gc()
        @test Array(x) == original
        expected = permutedims(original)[1:2:3, :]
        output = Array(y)
        @test output == expected
        output .= 999
        GC.gc()
        @test Array(y) == expected
        @test Array(x) == original
    end
end

@testset "indexing restrictions are explicit" begin
    for (Context, TensorType) in
        ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        ctx = Context()
        x = TensorType(ctx, reshape(collect(1.0:6), 2, 3))
        @test_throws ArgumentError x[1:0, :]
        @test_throws ArgumentError x[2:-1:1, :]
        @test_throws ArgumentError x[1:2]
        @test_throws BoundsError x[0, :]
        @test_throws BoundsError x[:, 4]
        v = TensorType(ctx, [1.0, 2, 3])
        @test_throws BoundsError v[[0, 2]]
        @test_throws BoundsError v[[1, 4]]
    end
end
