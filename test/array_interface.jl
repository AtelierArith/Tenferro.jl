using Test, Tenferro, Statistics, NNlib, Functors
using DifferentiationInterface, FiniteDiff

const BACKENDS = (AutoTenferro(), AutoTenferroTraced())
const CONTEXTS = ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
fd_gradient(f, x) = FiniteDiff.finite_difference_gradient(f, x)
values_of(r) = Array(r)

const A = [0.3 -1.2 2.1 0.7; -0.4 1.5 -2.6 0.9; 1.1 -0.8 0.2 -1.7]
const W = reshape(collect(1.0:12.0), 3, 4)

@testset "element-type casts" begin
    for (Context, TensorType) in CONTEXTS
        ctx = Context()
        x = TensorType(ctx, Float32.(A))
        x64 = TensorType(ctx, A)
        @test Float64.(x) isa TensorType{Float64,2}
        @test Array(Float64.(x)) == Float64.(Float32.(A))
        @test Array(Float32.(x64)) == Float32.(A)
        @test Array(ComplexF64.(x)) == ComplexF64.(Float32.(A))
        @test Array(Float32.(TensorType(ctx, Int32[1 0; -2 3]))) == Float32[1 0; -2 3]
        @test Array(Float64.(x .> 0)) == Float64.(A .> 0)
        @test Array(Int64.(TensorType(ctx, [true, false]))) == [1, 0]
        @test Array(trunc.(Int32, TensorType(ctx, [1.7, -2.7]))) == Int32[1, -2]
        @test Array(TensorType(ctx, [0.0, 2.0]) .!= 0) == [false, true]
        @test Array(float.(TensorType(ctx, [1, 2]))) == [1.0, 2.0]
        @test convert(TensorType{Float64}, x) isa TensorType{Float64,2}
        @test convert(TensorType{Float32}, x) === x
        @test convert(TensorType, x) === x
        @test_throws ArgumentError convert(TensorType{Float64,3}, x)
        z = TensorType(ctx, ComplexF64[1 + 2im, -3 - 4im])
        @test_throws (TensorType === EagerTensor ? InexactError : ArgumentError) Float64.(z)
        @test Array(real.(z)) == [1.0, -3.0]
        @test Array(imag.(z)) == [2.0, -4.0]
        @test Array(imag.(x64)) == zero(A)
        @test real.(x64) === x64
        # Promotions now cast explicitly, also from Float64 down.
        @test Array(x .+ x64) ≈ Float32.(A) .+ A
    end
    ctx = EagerContext()
    s = sum(EagerTensor(ctx, Float32[1, 2]))
    @test Float64.(s) isa Tenferro.EagerScalar{Float64}
    @test only(Float64.(s)) === 3.0
end

@testset "cast gradients" begin
    for backend in BACKENDS
        f32(a) = sum(Float64.(a) .^ 2)
        g = gradient(f32, backend, Float32.(A))
        @test eltype(g) === Float32
        @test g ≈ 2 .* Float32.(A)
        f64(a) = sum(Float32.(a) .* Float32.(W))
        g = gradient(f64, backend, A)
        @test eltype(g) === Float64
        @test g ≈ W
        fc(a) = sum(real.(ComplexF64.(a) .* (1 + 2im)) .+ imag.(ComplexF64.(a) .* (2 - 1im)))
        @test gradient(fc, backend, A) ≈ fill(0.0, size(A)) atol = 1e-12
        fc2(a) = sum(imag.(ComplexF64.(a) .* (0.5 + 2im)) .* W)
        @test gradient(fc2, backend, A) ≈ 2 .* W
        # A masked mean: the Bool -> float cast carries no cotangent.
        masked(a) = sum(a .* Float64.(a .> 0)) / sum(a .> 0)
        @test gradient(masked, backend, A) ≈ (A .> 0) ./ count(A .> 0)
        masked32(a) = sum(a .* Float32.(a .> 0)) / sum(a .> 0)
        @test gradient(masked32, backend, Float32.(A)) ≈ Float32.((A .> 0) ./ count(A .> 0))
    end
end

@testset "public constants and contexts" begin
    for (Context, TensorType) in CONTEXTS
        ctx = Context()
        x = TensorType(ctx, A)
        @test Tenferro.context(x) isa Context
        @test Tenferro.context(x).graph === x.graph
        c = Tenferro.constant(x, W)
        @test c isa TensorType{Float64,2}
        @test Array(c .+ x) == W .+ A
        y = TensorType(Tenferro.context(x), W)
        @test Array(x .* y) == A .* W
        k = Tenferro.constant(x, 2.0f0)
        @test k isa TensorType{Float32,0}
        @test_throws ArgumentError Tenferro.constant(x, Int16[1])
    end
    s = sum(EagerTensor(EagerContext(), [1.0]))
    @test Tenferro.context(s) isa EagerContext
    @test Array(Tenferro.constant(s, [1, 2])) == [1, 2]
    for backend in BACKENDS
        @test gradient(a -> sum(a .* Tenferro.constant(a, W)), backend, A) ≈ W
    end
end

@testset "Functors treats tensors as leaves" begin
    for (Context, TensorType) in CONTEXTS
        x = TensorType(Context(), A)
        y = TensorType(Context(), W)
        leaves = Functors.fleaves((w=x, layer=(b=y,)))
        @test length(leaves) == 2
        @test leaves[1] === x && leaves[2] === y
        doubled = fmap(t -> t .* 2, (w=x, b=y))
        @test Array(doubled.w) == 2A
    end
    s = sum(EagerTensor(EagerContext(), [1.0]))
    @test Functors.fleaves((s,)) == [s]
end

@testset "vec, repeat and elementwise constants" begin
    for (Context, TensorType) in CONTEXTS
        x = TensorType(Context(), A)
        @test Array(vec(x)) == vec(A)
        @test vec(vec(x)) isa TensorType{Float64,1}
        for args in ((2,), (2, 3), (1, 1, 2), (1,))
            @test Array(repeat(x, args...)) == repeat(A, args...)
        end
        for (inner, outer) in (((2, 1), (1, 3)), ((1, 2), nothing), (nothing, (2, 1, 2)), ((2, 2, 2), nothing))
            kw = (; (inner === nothing ? () : (; inner))..., (outer === nothing ? () : (; outer))...)
            @test Array(repeat(x; kw...)) == repeat(A; kw...)
        end
        @test Array(repeat(TensorType(Context(), [1, 2]), 2)) == [1, 2, 1, 2]
        @test_throws ArgumentError repeat(x, 0)
        @test_throws ArgumentError repeat(x; inner=(2,))
        @test Array(zero.(x)) == zero(A)
        @test Array(one.(x)) == one.(A)
        err = try
            similar(x)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("immutable", err.msg)
        @test_throws ArgumentError similar(x, Float32)
        @test_throws ArgumentError similar(x, Float32, (2, 2))
    end
    for backend in BACKENDS
        for f in (a -> sum(repeat(a, 2, 3) .* repeat(W, 2, 3) .^ 2),
                  a -> sum(repeat(a; inner=(2, 1), outer=(1, 2)) .* repeat(W; inner=(1, 2), outer=(2, 1))),
                  a -> sum(vec(a) .* vec(W)),
                  a -> sum(a .* one.(a) .+ zero.(a)))
            @test gradient(f, backend, A) ≈ fd_gradient(f, A) rtol = 1e-6
        end
    end
end

@testset "mapped reductions and n-ary broadcasts" begin
    for (Context, TensorType) in CONTEXTS
        x = TensorType(Context(), A)
        for f in (abs2, abs, exp), dims in (:, 1, 2, (1, 2))
            @test vec(Array(sum(f, x; dims))) ≈ vec([sum(f, A; dims)...])
            @test vec(Array(prod(f, x; dims))) ≈ vec([prod(f, A; dims)...])
            @test vec(Array(maximum(f, x; dims))) ≈ vec([maximum(f, A; dims)...])
            @test vec(Array(minimum(f, x; dims))) ≈ vec([minimum(f, A; dims)...])
            @test vec(Array(mean(f, x; dims))) ≈ vec([mean(f, A; dims)...])
        end
        @test only(maximum(abs, x; init=10.0)) == 10.0
        @test Array(max.(x, 0.5, W .- 5)) == max.(A, 0.5, W .- 5)
        @test Array(min.(x, x .* 2, 0.0, W)) == min.(A, A .* 2, 0.0, W)
        @test Array((+).(x, 1, W, x)) ≈ (+).(A, 1, W, A)
        @test Array((*).(x, 2, W)) ≈ (*).(A, 2, W)
        @test Array(muladd.(x, W, 1.0)) ≈ muladd.(A, W, 1.0)
        @test Array(muladd.(2.0, x, W)) ≈ muladd.(2.0, A, W)
        @test Array(fma.(x, x, W)) ≈ fma.(A, A, W)
        @test Array((>(0)).(x)) == (A .> 0)
        @test Array((-).(1.0, x) .+ (Base.Fix1(-, 1.0)).(x)) == 2 .* (1 .- A)
    end
    for backend in BACKENDS
        for f in (a -> sum(abs2, a), a -> sum(sum(exp, a; dims=1) .* W[1:1, :]), a -> mean(abs2, a; dims=2)[2, 1],
                  a -> maximum(abs, a), a -> sum(max.(a, 0.5, W .- 11)), a -> sum(muladd.(a, W, a)),
                  a -> sum(fma.(a, a, W)), a -> sum((+).(a, a, W, a) .* W), a -> prod(abs, a; dims=2)[1, 1])
            @test gradient(f, backend, A) ≈ fd_gradient(f, A) rtol = 1e-6
        end
    end
end

@testset "cat of tensors and plain arrays" begin
    for (Context, TensorType) in CONTEXTS
        x = TensorType(Context(), A)
        @test Array(hcat(x, ones(3, 1))) == hcat(A, ones(3, 1))
        @test Array(hcat(ones(3, 1), x)) == hcat(ones(3, 1), A)
        @test Array(vcat(ones(1, 4), x)) == vcat(ones(1, 4), A)
        @test Array(vcat(x, W, x)) == vcat(A, W, A)
        @test Array(cat(ones(1, 4), zeros(2, 4), x; dims=1)) == cat(ones(1, 4), zeros(2, 4), A; dims=1)
        @test Array(cat(x, W; dims=3)) == cat(A, W; dims=3)
        @test Array(hcat(x, ones(Float32, 3, 1))) == hcat(A, ones(3, 1))
        @test_throws DimensionMismatch hcat(x, ones(2, 1))
    end
    for backend in BACKENDS
        f(a) = sum(hcat(ones(3, 1), a, W) .* reshape(collect(1.0:27.0), 3, 9))
        @test gradient(f, backend, A) ≈ fd_gradient(f, A)
        g(a) = sum(vcat(W, a) .^ 2)
        @test gradient(g, backend, A) ≈ 2 .* A
    end
end

@testset "Bool and integer reductions" begin
    for (Context, TensorType) in CONTEXTS
        x = TensorType(Context(), A)
        m = x .> 0
        @test only(sum(m)) === sum(A .> 0)
        @test Array(sum(m; dims=1)) == sum(A .> 0; dims=1)
        @test only(count(m)) === count(A .> 0)
        @test Array(count(m; dims=2)) == count(A .> 0; dims=2)
        @test only(count(>(0), x)) == count(>(0), A)
        @test only(any(m)) === true
        @test only(all(m)) === false
        @test only(all(x .> -10)) === true
        @test only(any(x .> 10)) === false
        @test Array(any(m; dims=2)) == any(A .> 0; dims=2)
        @test Array(all(m; dims=1)) == all(A .> 0; dims=1)
        @test only(any(>(2), x)) === any(>(2), A)
        @test only(all(<(3), x)) === all(<(3), A)
        @test only(mean(m)) ≈ mean(A .> 0)
        @test only(mean(TensorType(Context(), [1, 2, 4]))) ≈ 7 / 3
        @test Array(mean(m; dims=1)) ≈ mean(A .> 0; dims=1)
        e = TensorType(Context(), Bool[])
        @test only(sum(e)) === 0
        @test only(any(e)) === false
        @test only(all(e)) === true
        @test_throws ArgumentError count(abs, x)
    end
    @test Bool(any(EagerTensor(EagerContext(), [false, true])))
end

@testset "NNlib gather and scatter" begin
    src = reshape(collect(1.0:24.0), 2, 3, 4)
    idxs = ([1, 3, 3, 2], [3 1; 2 2], [(1, 2), (3, 4), (1, 2)], [CartesianIndex(2, 1), CartesianIndex(3, 3)])
    for (Context, TensorType) in CONTEXTS
        x = TensorType(Context(), src)
        for idx in idxs
            @test Array(NNlib.gather(x, idx)) == NNlib.gather(src, idx)
        end
        @test Array(NNlib.gather(x, [1, 2], [2, 4])) == NNlib.gather(src, [1, 2], [2, 4])
        v = TensorType(Context(), [10.0, 20.0, 30.0])
        @test Array(NNlib.gather(v, [3, 1, 3])) == [30.0, 10.0, 30.0]
        @test_throws BoundsError NNlib.gather(x, [5])
        @test_throws BoundsError NNlib.gather(x, [(0, 1)])

        m = TensorType(Context(), A)
        for (op, idx, kw) in ((+, [2, 1, 1, 5], (;)), (+, [3, 1, 1, 3], (; init=1.0)), (mean, [2, 1, 1, 5], (;)),
                              (-, [1, 1, 2, 2], (;)), (+, [2, 2, 2, 2], (; dstsize=(3, 4))),
                              (-, [1, 2, 1, 2], (; init=2.0, dstsize=(3, 3))))
            @test Array(NNlib.scatter(op, m, idx; kw...)) ≈ NNlib.scatter(op, A, idx; kw...)
        end
        s3 = reshape(collect(1.0:12.0), 3, 2, 2)
        t3 = TensorType(Context(), s3)
        for idx in ([(1, 1) (2, 2); (1, 1) (1, 2)], [1 1; 2 3])
            @test Array(NNlib.scatter(+, t3, idx)) == NNlib.scatter(+, s3, idx)
        end
        @test Array(NNlib.scatter(+, TensorType(Context(), [10, 100, 1000]), [3, 1, 2])) == [100, 1000, 10]
        @test_throws ArgumentError NNlib.scatter(max, m, [1, 2, 3, 4])
        @test_throws ArgumentError NNlib.scatter(+, m, [1, 2])
    end
    for backend in BACKENDS
        for f in (a -> sum(NNlib.gather(a, [3, 1, 3, 2]) .* reshape(collect(1.0:12.0), 3, 4)),
                  a -> sum(NNlib.gather(a, [(1, 2), (3, 4), (1, 2)]) .^ 2),
                  a -> sum(NNlib.scatter(+, a, [2, 1, 2, 5]) .* reshape(collect(1.0:15.0), 3, 5)),
                  a -> sum(NNlib.scatter(mean, a, [2, 1, 2, 2]) .^ 2),
                  a -> sum(NNlib.scatter(-, a, [1, 1, 3, 1]; init=1.0) .* W[:, 1:3]))
            @test gradient(f, backend, A) ≈ fd_gradient(f, A) rtol = 1e-6
        end
    end
end
