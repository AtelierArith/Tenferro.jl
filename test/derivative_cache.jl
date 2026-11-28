using Test, Tenferro, DifferentiationInterface

function cache_stats()
    out = zeros(UInt64, 3)
    Tenferro.RustCall.result_to_exception(Tenferro.RustCrate.derivative_cache_stats(out))
    return (entries=Int(out[1]), hits=Int(out[2]), misses=Int(out[3]))
end

@testset "derivative programs are reused across calls" begin
    W, V, b = [1.0 2.0 0.5; -1.0 0.0 3.0], [0.5 -2.0], [0.25, -0.5]
    loss(p, x) = sum(abs2.(p.V * max.(p.W * x .+ p.b, 0.0)))
    p = (; W, V, b)
    g1 = gradient(loss, AutoTenferro(), p, Constant([1.0, 2.0, 3.0]))
    before = cache_stats()
    # Same function and shapes, new data: one reverse pass that hits the cache.
    g2 = gradient(loss, AutoTenferro(), p, Constant([-1.0, 0.5, 2.0]))
    after = cache_stats()
    @test after.misses == before.misses
    @test after.hits == before.hits + 1
    g3 = gradient(loss, AutoTenferro(), p, Constant([1.0, 2.0, 3.0]))
    @test g3 == g1
    @test g2 != g1
end

@testset "one reverse pass for all inputs" begin
    A, B = [1.0 2.0; 3.0 4.0], [0.5 -1.0; 2.0 0.0]
    for backend in (AutoTenferro(), AutoTenferroTraced())
        # Unused input and an input without cotangent flow get zero gradients.
        ga, gb, gc = gradient(t -> sum(t[1] * t[2]) + sum(sign.(t[3])), backend, (A, B, [1.0, -2.0]))
        @test ga ≈ ones(2, 2) * B'
        @test gb ≈ A' * ones(2, 2)
        @test gc == zeros(2)
        _, (gx,) = value_and_gradient(t -> sum(Float64.(t[1] .> 0)), backend, (A,))
        @test gx == zeros(2, 2)
        # Several seeds do not accumulate into each other.
        (p1, p2) = pullback(x -> x * x, backend, A, (ones(2, 2), 2ones(2, 2)))
        @test p2 ≈ 2p1
    end
end
