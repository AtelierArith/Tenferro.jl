using Test, Tenferro, DifferentiationInterface, NNlib

function traced_cache_stats()
    out = zeros(UInt64, 3)
    Tenferro.RustCall.result_to_exception(Tenferro.RustCrate.traced_cache_stats(out))
    return (entries=Int(out[1]), hits=Int(out[2]), misses=Int(out[3]))
end

const TRACED = AutoTenferroTraced()

@testset "traced programs are prepared once per structure" begin
    loss(p, x) = sum(abs2.(relu.(p.W * x .+ p.b)))
    W, b = [1.0 2.0 -0.5; -1.0 0.0 3.0], [0.25, -0.5]
    x1, x2 = [1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]
    v1, g1 = value_and_gradient(loss, TRACED, (; W, b), Constant(x1))
    before = traced_cache_stats()
    # Same structure and shapes, new data: the prepared program is reused...
    v2, g2 = value_and_gradient(loss, TRACED, (; W, b), Constant(x2))
    after = traced_cache_stats()
    @test after.hits == before.hits + 1
    @test after.misses == before.misses
    # ...and runs on the new data.
    e2 = value_and_gradient(loss, AutoTenferro(), (; W, b), Constant(x2))
    @test v2 ≈ e2[1]
    @test g2.W ≈ e2[2].W && g2.b ≈ e2[2].b
    @test g2.W != g1.W
    @test value_and_gradient(loss, TRACED, (; W, b), Constant(x1)) == (v1, g1)
    # `gradient` alone is another program, and it is cached too.
    @test gradient(loss, TRACED, (; W, b), Constant(x1)) == g1
    hits = traced_cache_stats().hits
    @test gradient(loss, TRACED, (; W, b), Constant(x2)) == g2
    @test traced_cache_stats().hits == hits + 1
    # New parameter values are inputs too.
    g3 = gradient(loss, TRACED, (; W=2W, b), Constant(x1))
    @test g3.W ≈ gradient(loss, AutoTenferro(), (; W=2W, b), Constant(x1)).W
    # Another shape is another program.
    misses = traced_cache_stats().misses
    gradient(loss, TRACED, (; W=ones(2, 4), b), Constant(ones(4)))
    @test traced_cache_stats().misses == misses + 1
end

@testset "numbers and arrays mixed into a traced graph are inputs of the cached program" begin
    for c in (2.0, -3.0, 0.5)
        @test gradient(a -> sum(a .* c), TRACED, [1.0, 2.0]) == [c, c]
        @test gradient(a -> sum(a .* [c, 2c]), TRACED, [1.0, 2.0]) == [c, 2c]
        @test value_and_gradient(a -> sum(a .^ 2) * c, TRACED, [1.0, 2.0]) == (5c, [2c, 4c])
    end
    # Values of nodes evaluated separately use the cache as well.
    for s in (1.0, 2.0)
        ctx = TracedContext()
        x = TracedTensor(ctx, [s, 2s])
        @test Array(x .* x) == [s^2, 4s^2]
        @test Array(sum(x)) == fill(3s)
        close(ctx)
    end
end

@testset "cached traced programs: pushforwards, convolutions and batches" begin
    w = reshape(collect(1.0:18.0) ./ 10, 3, 3, 1, 2)
    conv_loss(x) = sum(abs2.(NNlib.conv(x, w; pad=1)))
    for k in 1:3
        x = reshape(sin.(k .* (1:50)), 5, 5, 1, 2)
        @test gradient(conv_loss, TRACED, x) ≈ gradient(conv_loss, AutoTenferro(), x)
        d = cos.(k .* x)
        @test pushforward(conv_loss, TRACED, x, (d,))[1] ≈ pushforward(conv_loss, AutoTenferro(), x, (d,))[1]
    end
    # A batch of another size is another program; both stay correct.
    for n in (2, 3, 2)
        x = reshape(sin.(1:(25n)), 5, 5, 1, n)
        @test gradient(conv_loss, TRACED, x) ≈ gradient(conv_loss, AutoTenferro(), x)
    end
end

@testset "repeated traced graphs reuse their reverse pass and program, on new data" begin
    loss(p, x) = sum(abs2.(relu.(p.W * x .+ p.b)))
    for k in 1:4
        W, b, x = sin.(k .* [1.0 2.0 3.0; 4.0 5.0 6.0]), cos.(k .* [1.0, 2.0]), tan.(k .* [0.1, 0.2, 0.3])
        v, g = value_and_gradient(loss, TRACED, (; W, b), Constant(x))
        ev, eg = value_and_gradient(loss, AutoTenferro(), (; W, b), Constant(x))
        @test v ≈ ev && g.W ≈ eg.W && g.b ≈ eg.b
        # Seeds and tangents are data of each call as well.
        seeds = (k * 1.0, -2.0k)
        t = value_and_pullback(loss, TRACED, (; W, b), seeds, Constant(x))
        e = value_and_pullback(loss, AutoTenferro(), (; W, b), seeds, Constant(x))
        @test t[1] ≈ e[1]
        @test all(i -> t[2][i].W ≈ e[2][i].W && t[2][i].b ≈ e[2][i].b, 1:2)
        f(v) = sum(abs2.(v .* x))
        @test pushforward(f, TRACED, x, (k .* x,))[1] ≈ pushforward(f, AutoTenferro(), x, (k .* x,))[1]
    end
end
