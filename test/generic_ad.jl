using Test, Tenferro, DifferentiationInterface

@testset "element types and dimensions" begin
    for T in (Float32, Float64), backend in (AutoTenferro(), AutoTenferroTraced())
        x = reshape(T.(1:6), 2, 3)
        g = gradient(a -> sum(a * a), backend, x[:, 1:2])
        expected = ones(T, 2, 2) * x[:, 1:2]' + x[:, 1:2]' * ones(T, 2, 2)
        @test g isa Matrix{T}
        @test g ≈ expected
        for a in (fill(T(2)), T[1, 2, 3], reshape(T.(1:8), 2, 2, 2), @view(x[:, 2:3]), TypedTensor(x))
            g = gradient(z -> sum(z + z), backend, a)
            @test g isa Array{T,ndims(a)}
            @test g == fill(T(2), size(a))
        end
        a = T[1, 2, 3]
        ga, gb = gradient(z -> sum(z[1] + z[1]), backend, (a, copy(a)))
        @test ga == fill(T(2), 3)
        @test gb == zeros(T, 3)
    end
    for T in (Float32, Float64)
        a = T[1, 2, 3]
        ((da,),) = pullback(z -> z[1] + z[1], AutoTenferro(), (a,), (fill(T(3), 3),))
        @test da == fill(T(6), 3)
    end
end
