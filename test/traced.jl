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

@testset "traced JVP and VJP" begin
    backend = AutoTenferroTraced()
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = T[1 2; 3 4]; b = T[2 0; 1 3]
        da = T[0.1 0.2; -0.3 0.4]; db = T[-0.2 0.1; 0.3 0.2]
        w = T[0.2 0.4; 0.1 -0.3]
        if T <: Complex
            a .+= im; da .*= 1+im; w .*= 1-im
        end
        f(z) = z[1] * z[2]
        (dy, dy2) = pushforward(f, backend, (a,b), ((da,db), (da,zero(db))))
        @test dy ≈ da*b + a*db
        @test dy2 ≈ da*b
        ((ga,gb),) = pullback(f, backend, (a,b), (w,))
        @test ga ≈ w*b'
        @test gb ≈ a'*w
        @test real(sum(conj.(w).*dy)) ≈ real(sum(conj.(ga).*da) + sum(conj.(gb).*db))
        ((ga,gb),) = pullback(z -> z[1], backend, (a,b), (w,))
        @test ga ≈ w
        @test gb == zero(b)
        (ds,) = pushforward(sum, backend, a, (da,))
        @test ds ≈ sum(da)
        (gs,) = pullback(sum, backend, a, (one(T),))
        @test gs == ones(T, size(a))
    end
    x = ones(2,2)
    @test_throws DimensionMismatch pullback(identity, backend, x, (ones(3),))
    @test_throws ArgumentError pullback(identity, backend, x, (ones(Float32,2,2),))
    @test_throws DimensionMismatch pushforward(identity, backend, x, (ones(3),))
    @test_throws ArgumentError pushforward(identity, backend, x, (ones(Float32,2,2),))
    @test only(pushforward(sum, backend, Float64[], (Float64[],))) == 0.0
    @test only(pullback(sum, backend, Float64[], (1.0,))) == Float64[]
end
