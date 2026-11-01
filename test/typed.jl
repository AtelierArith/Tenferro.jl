using Test, Tenferro

@testset "TypedTensor array interoperability" begin
    for T in (Float32, Float64, Int32, Int64, ComplexF32, ComplexF64), dims in ((), (4,), (2, 3), (2, 1, 3), (0, 2))
        a = reshape(T.(1:prod(dims; init=1)), dims)
        x = TypedTensor(a)
        @test x isa AbstractArray{T,length(dims)}
        @test size(x) == dims
        @test Array(x) == a
        @test collect(x) == a
        if !isempty(a)
            x[1] = T(7)
            @test x[1] == T(7)
            @test a[1] == T(1)
        end
    end
    a = reshape(collect(1.0:12.0), 3, 4)
    v = @view a[1:2:3, 2:3]
    x = TypedTensor(v)
    @test Array(x) == v
    @test x[2, 2] == v[2, 2]
    x[2, 2] = 99
    @test x[2, 2] == 99
    @test v[2, 2] != 99
    @test_throws BoundsError x[100]
    @test_throws ArgumentError TypedTensor(ones(Int16, 2))
    complex_values = ComplexF64[1 + 2im 3 - 4im; -2im 1 + 0im]
    cx = TypedTensor(complex_values)
    @test Array(cx) == complex_values
    cx[1, 2] = -7 + 8im
    @test cx[1, 2] == -7 + 8im
    @test Array(TypedTensor(fill(3.0)))[] == 3.0
    ctx = EagerContext()
    @test Array(EagerTensor(ctx, x)) == Array(x)
    @test Array(TracedTensor(TracedContext(), x)) == Array(x)
end
