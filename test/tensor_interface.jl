using Test, Tenferro, Statistics
using DifferentiationInterface, ADTypes, FiniteDiff, ChainRulesCore

const BACKENDS = (AutoTenferro(), AutoTenferroTraced())
const CONTEXTS = ((EagerContext, EagerTensor), (TracedContext, TracedTensor))

fd_gradient(f, x) = gradient(f, AutoFiniteDiff(), x)

@testset "scalar value extraction" begin
    ctx = EagerContext()
    s = sum(EagerTensor(ctx, Float32[1, 2, 3]))
    @test s[] === 6.0f0
    @test only(s) === 6.0f0
    @test Float64(s) === 6.0
    @test convert(Float32, s) === 6.0f0
    @test Array(s) isa Array{Float32,0}
    @test Array(s)[] == 6
    @test size(s) == () && ndims(s) == 0 && length(s) == 1
    t = sum(TracedTensor(TracedContext(), [1.0, 2.0]))
    @test only(t) === 3.0
    @test Float32(t) === 3.0f0
end

@testset "ADTypes and DifferentiationInterface metadata" begin
    for backend in BACKENDS
        @test ADTypes.mode(backend) isa ADTypes.ReverseMode
        @test DifferentiationInterface.inplace_support(backend) == DifferentiationInterface.InPlaceNotSupported()
    end
end

@testset "value_and_gradient, value_and_pullback and value_and_pushforward" begin
    x = [1.0, -2.0, 3.0]
    f(a) = sum(a .* a .* a)
    for backend in BACKENDS
        value, grad = value_and_gradient(f, backend, x)
        @test value ≈ sum(x .^ 3)
        @test grad ≈ 3 .* x .^ 2
        value, (back,) = value_and_pullback(a -> a .* a, backend, x, ([1.0, 2.0, 3.0],))
        @test value ≈ x .^ 2
        @test back ≈ 2 .* x .* [1.0, 2.0, 3.0]
        value, (back,) = value_and_pullback(f, backend, x, (2.0,))
        @test value ≈ sum(x .^ 3)
        @test back ≈ 6 .* x .^ 2
        value, (dy,) = value_and_pushforward(a -> a .* a, backend, x, ([1.0, 0.0, 1.0],))
        @test value ≈ x .^ 2
        @test dy ≈ 2 .* x .* [1.0, 0.0, 1.0]
        @test_throws ArgumentError gradient(identity, backend, x)
    end
end

@testset "Constant contexts" begin
    x = [1.0, 2.0, 3.0]
    c = [4.0, -5.0, 6.0]
    for backend in BACKENDS
        @test gradient((a, k) -> sum(a .* k), backend, x, Constant(c)) ≈ c
        value, grad = value_and_gradient((a, k, s) -> sum(a .* k) * s, backend, x, Constant(c), Constant(2.0))
        @test value ≈ 2 * sum(x .* c)
        @test grad ≈ 2 .* c
        (back,) = pullback((a, k) -> a .* k, backend, x, ([1.0, 1.0, 1.0],), Constant(c))
        @test back ≈ c
        (dy,) = pushforward((a, k) -> a .* k, backend, x, ([1.0, 0.0, 2.0],), Constant(c))
        @test dy ≈ c .* [1.0, 0.0, 2.0]
    end
end

@testset "gradients of nested parameter structures" begin
    w, b = [1.0 2.0; 3.0 4.0], [0.5, -1.0]
    params = (layer_1 = (weight = w, bias = b), layer_2 = NamedTuple(), scale = (Float32[2],))
    loss(p) = sum(p.layer_1.weight * p.layer_1.bias) + sum(p.layer_1.bias .* p.layer_1.bias)
    for backend in BACKENDS
        g = gradient(p -> sum(p.w) + sum(p.b), backend, (w = rand(2), b = rand(3)))
        @test g == (w = ones(2), b = ones(3))
        g = gradient(loss, backend, params)
        @test g.layer_1.weight ≈ ones(2) * b'
        @test g.layer_1.bias ≈ w' * ones(2) .+ 2 .* b
        @test g.layer_2 === NamedTuple()
        @test g.scale == (zeros(Float32, 1),)
        value, g = value_and_gradient(p -> sum(p[1]) * p[2], backend, ([1.0, 2.0], 3))
        @test value == 9.0
        @test g == ([3.0, 3.0], nothing)
        ((gw, gb),) = pullback(p -> p.w * p.b, backend, (w = w, b = b), ([1.0, 1.0],))
        @test gw ≈ ones(2) * b'
        @test gb ≈ w' * ones(2)
        (dy,) = pushforward(p -> p.w * p.b, backend, (w = w, b = b), ((w = ones(2, 2), b = zeros(2)),))
        @test dy ≈ ones(2, 2) * b
        @test_throws ArgumentError pushforward(p -> p.w * p.b, backend, (w = w, b = b), ((w = ones(2, 2),),))
        dense(p, x) = sum(abs2.(p.weight * x .+ p.bias))
        value, g = value_and_gradient(dense, backend, (weight = w, bias = b), Constant([1.0, 2.0]))
        r = w * [1.0, 2.0] .+ b
        @test value ≈ sum(abs2, r)
        @test g.weight ≈ 2 .* r * [1.0, 2.0]'
        @test g.bias ≈ 2 .* r
    end
end

@testset "comparisons and ifelse" begin
    for (Context, Tensor) in CONTEXTS
        ctx = Context()
        a, b = Float32[1, -2, 3], Float32[0, -2, 5]
        x, y = Tensor(ctx, a), Tensor(ctx, b)
        for op in (>, <, >=, <=, ==, !=)
            @test Array(op.(x, y)) == op.(a, b)
            @test Array(op.(x, 0.0f0)) == op.(a, 0.0f0)
            @test Array(op.(0, x)) == op.(0, a)
            @test Array(op.(x, b)) == op.(a, b)
            @test eltype(op.(x, y)) === Bool
        end
        @test Array(.!(x .> 0)) == .!(a .> 0)
        @test Array(ifelse.(x .> 0, x, y)) == ifelse.(a .> 0, a, b)
        @test Array(ifelse.(x .> 0, 1.0f0, y)) == ifelse.(a .> 0, 1.0f0, b)
        @test Array(ifelse.(a .> 0, x, b)) == ifelse.(a .> 0, a, b)
        @test Array(x .* (a .+ 1)) == a .* (a .+ 1)
        @test Array(x .> (a .- 1)) == (a .> (a .- 1))
        @test Array(ifelse.([true, false, true], 0.0f0, x)) == ifelse.([true, false, true], 0.0f0, a)
    end
    x = [1.5, -2.0, 0.5, -0.25]
    leaky(a) = sum(ifelse.(a .> 0, a .* a, 0.1 .* a))
    clipped(a) = sum(ifelse.(a .>= 0.5, a, [1.0, 2.0, 3.0, 4.0]) .* a)
    for backend in BACKENDS
        @test gradient(leaky, backend, x) ≈ ifelse.(x .> 0, 2 .* x, 0.1)
        @test gradient(clipped, backend, x) ≈ fd_gradient(clipped, x) rtol = 1e-6
        @test gradient(a -> sum(a .* (a .> 0)), backend, x) ≈ fd_gradient(a -> sum(a .* (a .> 0)), x) rtol = 1e-6
    end
end

@testset "elementwise helpers" begin
    x = [1.0, -2.0, 4.0]
    for (Context, Tensor) in CONTEXTS
        ctx = Context()
        t = Tensor(ctx, x)
        @test Array(abs2.(t)) ≈ abs2.(x)
        @test Array(inv.(t)) ≈ inv.(x)
        @test Array(zero(t)) == zero(x)
        @test Array(copy(t)) == x
        z = Tensor(ctx, ComplexF64[1 + 2im, -1im])
        @test Array(abs2.(z)) ≈ [5.0, 1.0]
    end
    for backend in BACKENDS
        @test gradient(a -> sum(abs2.(a)), backend, x) ≈ 2 .* x
        @test gradient(a -> sum(inv.(a)), backend, x) ≈ -1 ./ x .^ 2
        @test gradient(a -> sum(a .+ zero(a)), backend, x) == ones(3)
        @test gradient(a -> sum(copy(a) .* a), backend, x) ≈ 2 .* x
    end
    @test zero(sum(EagerTensor(EagerContext(), x)))[] == 0.0
end

@testset "shape helpers" begin
    data = reshape(collect(1.0:12), 2, 1, 6)
    for (Context, Tensor) in CONTEXTS
        t = Tensor(Context(), data)
        @test Array(dropdims(t; dims = 2)) == dropdims(data; dims = 2)
        @test_throws ArgumentError dropdims(t; dims = 1)
        @test Array(selectdim(t, 3, 4)) == selectdim(data, 3, 4)
        @test Array(selectdim(t, 3, 2:3)) == selectdim(data, 3, 2:3)
        slices = eachslice(t; dims = 3)
        @test length(slices) == 6
        @test all(i -> Array(slices[i]) == data[:, :, i], 1:6)
        slices = eachslice(t; dims = (1, 3))
        @test size(slices) == (2, 6)
        @test Array(slices[2, 5]) == data[2, :, 5]
        @test size(eachslice(t; dims = 3, drop = false)) == (1, 1, 6)
        ctx = Context()
        v, w = Tensor(ctx, [1.0, 2.0]), Tensor(ctx, [3.0, 4.0])
        m = Tensor(ctx, [1.0 2.0; 3.0 4.0])
        @test vcat(v, w) isa Tensor
        @test Array(vcat(v, w)) == [1.0, 2.0, 3.0, 4.0]
        @test Array(hcat(v, w)) == [1.0 3.0; 2.0 4.0]
        @test Array([m; m]) == [[1.0 2.0; 3.0 4.0]; [1.0 2.0; 3.0 4.0]]
        @test Array([m v]) == [1.0 2.0 1.0; 3.0 4.0 2.0]
        @test Array(cat(v, w; dims = 3)) == cat([1.0, 2.0], [3.0, 4.0]; dims = 3)
    end
    for backend in BACKENDS
        f(a) = sum(hcat(a, 2 .* a) * [1.0 2.0; 3.0 4.0]) + sum(vcat(a, a .* a))
        x = [1.0, -2.0]
        @test gradient(f, backend, x) ≈ fd_gradient(f, x) rtol = 1e-6
        g(a) = sum(abs2.(dropdims(a; dims = 2))) + sum(selectdim(a, 3, 2)) + reduce(+, map(sum, eachslice(a; dims = 3)))
        @test gradient(g, backend, data) ≈ fd_gradient(g, data) rtol = 1e-6
    end
end

@testset "integer-vector indexing" begin
    data = reshape(collect(1.0:12), 3, 4)
    for (Context, Tensor) in CONTEXTS
        t = Tensor(Context(), data)
        @test Array(t[:, [1, 3]]) == data[:, [1, 3]]
        @test Array(t[[3, 1, 1], 2:4]) == data[[3, 1, 1], 2:4]
        @test Array(t[[2, 3], [4, 1]]) == data[[2, 3], [4, 1]]
        @test Array(t[2, [1, 4]]) == data[2, [1, 4]]
        @test Array(t[[true, false, true], :]) == data[[true, false, true], :]
        @test Array(t[:]) == vec(data)
        @test_throws BoundsError t[:, [0, 1]]
        @test_throws BoundsError t[[1, 4], :]
        v = Tensor(Context(), [1.0, 2.0, 3.0])
        @test Array(v[[3, 1]]) == [3.0, 1.0]
        @test Array(v[[true, false, true]]) == [1.0, 3.0]
    end
    embedding(w) = sum(abs2.(w[:, [1, 3, 3]]))
    for backend in BACKENDS
        @test gradient(embedding, backend, data) ≈ fd_gradient(embedding, data) rtol = 1e-6
    end
    @test only(Array(EagerTensor(EagerContext(), [1.0, 2.0])[2:2])) == 2.0
    @test EagerTensor(EagerContext(), [1.0, 2.0])[2][] == 2.0
end

@testset "var and std" begin
    data = [1.0 2.0 -1.0; 4.0 3.0 0.5]
    for (Context, Tensor) in CONTEXTS
        t = Tensor(Context(), data)
        for corrected in (true, false), dims in (1, 2, (1, 2))
            @test Array(var(t; dims = dims, corrected = corrected)) ≈ var(data; dims = dims, corrected = corrected)
            @test Array(std(t; dims = dims, corrected = corrected)) ≈ std(data; dims = dims, corrected = corrected)
        end
        @test only(var(t)) ≈ var(data)
        @test only(std(t; corrected = false)) ≈ std(data; corrected = false)
        @test Array(var(t; dims = 2, mean = mean(data; dims = 2))) ≈ var(data; dims = 2)
    end
    loss(a) = sum(var(a; dims = 1)) + std(a; corrected = false) + sum(std(a; dims = 2))
    for backend in BACKENDS
        @test gradient(loss, backend, data) ≈ fd_gradient(loss, data) rtol = 1e-5
    end
end

@testset "stopgrad" begin
    x = [1.0, 2.0, -3.0]
    for backend in BACKENDS
        @test gradient(a -> sum(a .* Tenferro.stopgrad(a)), backend, x) ≈ x
        @test gradient(a -> sum(Tenferro.stopgrad(a)) + sum(a), backend, x) == ones(3)
        @test gradient(a -> sum(a .* ChainRulesCore.ignore_derivatives(a)), backend, x) ≈ x
    end
    ctx = EagerContext()
    s = sum(EagerTensor(ctx, x))
    @test Tenferro.stopgrad(s)[] == sum(x)
    @test Tenferro.stopgrad((1, [2.0])) == (1, [2.0])
    @test gradient(a -> sum(a) * Tenferro.stopgrad(sum(a)), AutoTenferro(), x) ≈ fill(sum(x), 3)
end
