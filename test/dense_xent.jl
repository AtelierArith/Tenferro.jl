using Test, Tenferro, NNlib, Statistics, LinearAlgebra
using DifferentiationInterface, FiniteDiff
using Tenferro: RustCall, RustCrate, _rust_call
const DI = DifferentiationInterface

const BACKENDS = (AutoTenferro(), AutoTenferroTraced())
const CONTEXTS = (EagerContext, TracedContext)
tensor(ctx, a) = ctx isa EagerContext ? EagerTensor(ctx, a) : TracedTensor(ctx, a)
tensor(ctx, ::Nothing) = nothing

# Deterministic test data.
pseudo(::Type{T}, dims...; k=1) where {T} = T.(reshape(sin.(k .* (1:prod(dims)) .+ 0.3k), dims))

# The dense layer and its derivatives on plain arrays, in Float64.
pre(W, x, b) = W * x .+ (b === nothing ? 0.0 : b)
mask(σ, z) = σ === relu ? Float64.(z .> 0) : ones(size(z))
function dense_reference(σ, W, x, b, seed)
    W, x = Float64.(W), Float64.(x)
    b = b === nothing ? nothing : Float64.(b)
    z = pre(W, x, b)
    c = mask(σ, z) .* seed
    gb = ndims(c) == 1 ? c : vec(sum(c; dims=2))
    return (y=σ.(z), W=c * x', x=W' * c, b=gb)
end
dense_tangent(σ, W, x, b, dW, dx, db) =
    mask(σ, pre(W, x, b)) .* (dW * x .+ W * dx .+ (db === nothing ? 0.0 : db))

tol(::Type{Float32}) = (rtol=2e-4, atol=1e-5)
tol(::Type{Float64}) = (rtol=1e-10, atol=1e-12)

# Derivative of node `y` with respect to the leaf `x` as a tensor of the same context, so it
# can be differentiated again: forward (tangent `seed` of `x`) or reverse (cotangent `seed`).
function derivative(y, x, seed, forward::Bool)
    g = y.graph
    id = if y isa EagerTensor
        RustCall.result_to_exception(_rust_call(g, forward ? RustCrate.jvp_node : RustCrate.vjp_node, UInt(y.id), UInt(x.id), UInt(seed.id)))
    else
        RustCall.result_to_exception(_rust_call(g, RustCrate.traced_derivative, UInt(y.id), UInt(x.id), UInt(seed.id), forward))
    end
    id >= 0 || return nothing
    return Tenferro._derivative_tensor(Tenferro.context(y), eltype(y), id, forward ? y.dims : x.dims)
end
scalar_tensor(s::EagerScalar) = Tenferro._scalar_tensor_value(s)
scalar_tensor(s) = s

@testset "dense values for every activation, bias and input shape" begin
    for T in (Float32, Float64), σ in (identity, relu, tanh, sigmoid), bias in (false, true), vector in (false, true)
        W = pseudo(T, 5, 7; k=1)
        x = vector ? pseudo(T, 7; k=2) : pseudo(T, 7, 3; k=2)
        b = bias ? pseudo(T, 5; k=3) : nothing
        expected = σ.(pre(Float64.(W), Float64.(x), b === nothing ? nothing : Float64.(b)))
        for C in CONTEXTS
            ctx = C()
            # Every operand a tensor, and the weight alone a tensor.
            for args in ((tensor(ctx, W), tensor(ctx, x), tensor(ctx, b)), (tensor(ctx, W), x, b))
                y = Tenferro.dense(σ, args...)
                @test size(y) == size(expected)
                @test Array(y) ≈ expected rtol = tol(T).rtol atol = tol(T).atol
            end
            @test Array(Tenferro.dense(σ, W, tensor(ctx, x), b)) ≈ expected rtol = tol(T).rtol atol = tol(T).atol
            close(ctx)
        end
    end
end

@testset "dense gradients, pullbacks and pushforwards" begin
    for T in (Float32, Float64), σ in (identity, relu), bias in (false, true), vector in (false, true)
        W = pseudo(T, 5, 7; k=1)
        x = vector ? pseudo(T, 7; k=2) : pseudo(T, 7, 3; k=2)
        b = bias ? pseudo(T, 5; k=3) : nothing
        # No pre-activation sits near the ReLU kink, so finite differences are reliable.
        @test minimum(abs, pre(Float64.(W), Float64.(x), b === nothing ? nothing : Float64.(b))) > 1e-3
        w = pseudo(T, (vector ? (5,) : (5, 3))...; k=4)
        p = bias ? (; W, x, b) : (; W, x)
        layer(q) = Tenferro.dense(σ, q.W, q.x, get(q, :b, nothing))
        loss(q) = sum(layer(q) .* w)
        ref = dense_reference(σ, W, x, b, Float64.(w))
        dp = map(v -> pseudo(T, size(v)...; k=5), p)
        jvp_ref = dense_tangent(σ, Float64.(W), Float64.(x), b === nothing ? nothing : Float64.(b), Float64.(dp.W), Float64.(dp.x), get(dp, :b, nothing))
        for backend in BACKENDS
            value, g = value_and_gradient(loss, backend, p)
            @test value ≈ sum(ref.y .* w) rtol = tol(T).rtol
            @test g.W isa Matrix{T} && g.x isa Array{T,ndims(x)}
            @test g.W ≈ ref.W rtol = tol(T).rtol atol = tol(T).atol
            @test g.x ≈ ref.x rtol = tol(T).rtol atol = tol(T).atol
            bias && @test g.b ≈ ref.b rtol = tol(T).rtol atol = tol(T).atol
            if T === Float64
                @test g.W ≈ FiniteDiff.finite_difference_gradient(W -> sum(σ.(pre(W, x, b)) .* w), W) rtol = 1e-6
            end
            # A non-unit cotangent of the layer output, and a pushforward of all operands.
            (pb,) = pullback(layer, backend, p, (w,))
            @test pb.W ≈ ref.W rtol = tol(T).rtol atol = tol(T).atol
            @test pb.x ≈ ref.x rtol = tol(T).rtol atol = tol(T).atol
            (jvp,) = pushforward(layer, backend, p, (dp,))
            @test size(jvp) == size(jvp_ref)
            @test jvp ≈ jvp_ref rtol = tol(T).rtol atol = tol(T).atol
        end
    end
end

@testset "dense composes other activations and element types" begin
    W, x, b = pseudo(Float64, 4, 3; k=1), pseudo(Float64, 3, 2; k=2), pseudo(Float64, 4; k=3)
    for backend in BACKENDS, σ in (tanh, sigmoid, gelu, softplus)
        loss(q) = sum(abs2.(Tenferro.dense(σ, q.W, x, q.b)))
        g = gradient(loss, backend, (; W, b))
        @test g.W ≈ FiniteDiff.finite_difference_gradient(W -> sum(abs2.(σ.(W * x .+ b))), W) rtol = 1e-6
        @test g.b ≈ FiniteDiff.finite_difference_gradient(b -> sum(abs2.(σ.(W * x .+ b))), b) rtol = 1e-6
    end
    for C in CONTEXTS
        ctx = C()
        # Mixed element types and complex tensors take the composed path too.
        @test Array(Tenferro.dense(relu, tensor(ctx, Float32.(W)), x, b)) ≈ relu.(W * x .+ b) rtol = 1e-6
        Wc, xc = complex.(W, 0.5W), complex.(x, -x)
        @test Array(Tenferro.dense(identity, tensor(ctx, Wc), tensor(ctx, xc), complex.(b))) ≈ Wc * xc .+ b
        close(ctx)
    end
    ctx = EagerContext()
    Wt = EagerTensor(ctx, W)
    @test_throws DimensionMismatch Tenferro.dense(relu, Wt, ones(4, 2))
    @test_throws DimensionMismatch Tenferro.dense(relu, Wt, x, ones(3))
    @test_throws ArgumentError Tenferro.dense(relu, Wt, x, ones(4, 1))
    @test_throws ArgumentError Tenferro.dense(relu, Wt, ones(3, 2, 2))
    @test_throws ArgumentError Tenferro.dense(relu, W, x)
    close(ctx)
end

@testset "dense derivatives of second order stay native" begin
    T = Float64
    W, x, b = pseudo(T, 5, 7; k=1), pseudo(T, 7, 3; k=2), pseudo(T, 5; k=3)
    dW, dx = pseudo(T, 5, 7; k=6), pseudo(T, 7, 3; k=7)
    # Gradient of `sum(abs2.(dense(σ, W, x, b)))` with respect to `W` as a plain function.
    gradW(σ, W, x) = (z = pre(W, x, b); (mask(σ, z) .* 2σ.(z)) * x')
    h = 1e-6
    for C in CONTEXTS, σ in (identity, relu)
        ctx = C()
        Wt, xt, bt = tensor(ctx, W), tensor(ctx, x), tensor(ctx, b)
        L = scalar_tensor(sum(abs2.(Tenferro.dense(σ, Wt, xt, bt))))
        gW = derivative(L, Wt, tensor(ctx, fill(1.0)), false)
        @test Array(gW) ≈ gradW(σ, W, x) rtol = 1e-10
        # Forward over reverse: Hessian-vector products along W and along x.
        @test Array(derivative(gW, Wt, tensor(ctx, dW), true)) ≈ (gradW(σ, W .+ h .* dW, x) .- gradW(σ, W .- h .* dW, x)) ./ 2h rtol = 1e-6
        @test Array(derivative(gW, xt, tensor(ctx, dx), true)) ≈ (gradW(σ, W, x .+ h .* dx) .- gradW(σ, W, x .- h .* dx)) ./ 2h rtol = 1e-6
        # Reverse over reverse: the gradient of <gW, R> with respect to x.
        R = pseudo(T, 5, 7; k=8)
        inner = scalar_tensor(sum(gW .* tensor(ctx, R)))
        gx = derivative(inner, xt, tensor(ctx, fill(1.0)), false)
        @test Array(gx) ≈ FiniteDiff.finite_difference_gradient(x -> sum(gradW(σ, W, x) .* R), x) rtol = 1e-6
        close(ctx)
    end
end

@testset "dense on BLAS and on the built-in GEMM" begin
    # Odd sizes, a single row, column and inner dimension.
    for T in (Float32, Float64), (o, i, n) in ((33, 65, 17), (1, 9, 4), (6, 1, 5), (7, 8, 1))
        W, x, b = pseudo(T, o, i; k=1), pseudo(T, i, n; k=2), pseudo(T, o; k=3)
        w = pseudo(T, o, n; k=4)
        loss(q) = sum(Tenferro.dense(relu, q.W, q.x, q.b) .* w)
        ref = dense_reference(relu, W, x, b, Float64.(w))
        results = map((true, false)) do blas
            previous = Tenferro._use_blas_gemm(blas)
            try
                map(backend -> value_and_gradient(loss, backend, (; W, x, b)), BACKENDS)
            finally
                Tenferro._use_blas_gemm(previous)
            end
        end
        for result in results, (value, g) in result
            @test value ≈ sum(ref.y .* w) rtol = tol(T).rtol
            @test g.W ≈ ref.W rtol = tol(T).rtol atol = tol(T).atol
            @test g.x ≈ ref.x rtol = tol(T).rtol atol = tol(T).atol
            @test g.b ≈ ref.b rtol = tol(T).rtol atol = tol(T).atol
        end
    end
    # BLAS is registered when the package loads.
    @test Tenferro._use_blas_gemm(true)
end

xent_reference(ŷ, y, agg) = agg(-sum(Float64.(y) .* NNlib.logsoftmax(Float64.(ŷ); dims=1); dims=1))

@testset "logit cross-entropy values and derivatives" begin
    for T in (Float32, Float64), agg in (mean, sum), dims in ((4, 3), (4, 2, 3))
        ŷ = 3 .* pseudo(T, dims...; k=1)
        y = T.(abs.(pseudo(T, dims...; k=2)))
        y = y ./ sum(y; dims=1)
        expected = xent_reference(ŷ, y, agg)
        for C in CONTEXTS
            ctx = C()
            for target in (y, tensor(ctx, y))
                loss = Tenferro.logitcrossentropy(tensor(ctx, ŷ), target; agg)
                @test (C === EagerContext ? loss[] : only(Array(loss))) ≈ expected rtol = tol(T).rtol
            end
            close(ctx)
        end
        # The gradient with respect to the logits, and to the targets as an input.
        f(q) = Tenferro.logitcrossentropy(q.ŷ, q.y; agg)
        ref(ŷ, y) = agg(-sum(y .* NNlib.logsoftmax(ŷ; dims=1); dims=1))
        ŷ64, y64 = Float64.(ŷ), Float64.(y)
        gŷ = FiniteDiff.finite_difference_gradient(a -> ref(a, y64), ŷ64)
        gy = FiniteDiff.finite_difference_gradient(t -> ref(ŷ64, t), y64)
        dŷ, dy = pseudo(T, dims...; k=3), pseudo(T, dims...; k=4)
        jvp_ref = sum(gŷ .* dŷ) + sum(gy .* dy)
        frtol = T === Float32 ? 1e-4 : 1e-6
        for backend in BACKENDS
            value, g = value_and_gradient(f, backend, (; ŷ, y))
            @test value ≈ expected rtol = tol(T).rtol
            @test g.ŷ isa Array{T,length(dims)}
            @test g.ŷ ≈ gŷ rtol = frtol atol = 1e-7
            @test g.y ≈ gy rtol = frtol atol = 1e-7
            @test gradient(a -> Tenferro.logitcrossentropy(a, y; agg), backend, ŷ) ≈ gŷ rtol = frtol atol = 1e-7
            (jvp,) = pushforward(f, backend, (; ŷ, y), ((; ŷ=dŷ, y=dy),))
            @test jvp ≈ jvp_ref rtol = frtol
            # A non-unit cotangent of the loss scales the gradient.
            (pb,) = pullback(f, backend, (; ŷ, y), (T(2.5),))
            @test pb.ŷ ≈ 2.5 .* gŷ rtol = frtol atol = 1e-6
        end
    end
end

@testset "logit cross-entropy matches the composed loss and is stable" begin
    ŷ = [1000.0 -3.0; 0.0 2.0; -1000.0 1.0]
    y = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    for backend in BACKENDS
        value, g = value_and_gradient(a -> Tenferro.logitcrossentropy(a, y), backend, ŷ)
        @test isfinite(value) && all(isfinite, g)
        @test value ≈ xent_reference(ŷ, y, mean)
        composed(a) = -sum(y .* NNlib.logsoftmax(a; dims=1)) / 2
        @test g ≈ gradient(composed, backend, ŷ) atol = 1e-12
    end
end

@testset "logit cross-entropy errors" begin
    ctx = EagerContext()
    ŷ = EagerTensor(ctx, pseudo(Float64, 3, 2))
    @test_throws ArgumentError Tenferro.logitcrossentropy(ŷ, ones(3, 2); agg=maximum)
    @test_throws DimensionMismatch Tenferro.logitcrossentropy(ŷ, ones(3, 3))
    @test_throws ArgumentError Tenferro.logitcrossentropy(ŷ, EagerTensor(ctx, ones(Float32, 3, 2)))
    @test_throws ArgumentError Tenferro.logitcrossentropy(EagerTensor(ctx, complex.(ones(3, 2))), ones(3, 2))
    close(ctx)
    # Second derivatives through the softmax are refused, not computed wrongly.
    for C in CONTEXTS
        ctx = C()
        a = tensor(ctx, pseudo(Float64, 3, 2))
        L = scalar_tensor(Tenferro.logitcrossentropy(a, [1.0 0.0; 0.0 1.0; 0.0 0.0]))
        g = derivative(L, a, tensor(ctx, fill(1.0)), false)
        @test Array(g) ≈ FiniteDiff.finite_difference_gradient(a -> xent_reference(a, [1.0 0.0; 0.0 1.0; 0.0 0.0], mean), pseudo(Float64, 3, 2)) rtol = 1e-6
        err = try
            Array(derivative(g, a, tensor(ctx, ones(3, 2)), true))
            nothing
        catch e
            e
        end
        # The traced backend reports the refusing family without the rule's message.
        @test err !== nothing
        @test occursin("tenferro-julia.xent", sprint(showerror, err))
        @test C === TracedContext || occursin("not implemented", sprint(showerror, err))
        close(ctx)
    end
end

@testset "boundary copies and gradient copies" begin
    arrays = (Float32[1 -2; 3 4.5], [1.0 2.0 3.0], Int32[1, -2, 3], Int64[4 5; 6 7], [true false; false true],
        ComplexF32[1+2im 3-im], ComplexF64[1+2im, -3im])
    for C in CONTEXTS, a in arrays
        ctx = C()
        t = tensor(ctx, a)
        @test Array(t) == a
        @test Array(t) isa Array{eltype(a),ndims(a)}
        close(ctx)
    end
    A = reshape(collect(1.0:12.0), 3, 4)
    for C in CONTEXTS
        ctx = C()
        t = tensor(ctx, A)
        # Values that are not compact column-major views.
        @test Array(permutedims(t, (2, 1))) == permutedims(A, (2, 1))
        @test Array(t[2:3, 2:4]) == A[2:3, 2:4]
        @test Array(t[:, 2]) == A[:, 2]
        @test Array(t .> 6.0) == (A .> 6.0)
        @test Array(t .+ zeros(3, 1)) == A
        close(ctx)
    end
    for backend in BACKENDS
        # Broadcast, transposed, sliced and complex gradients.
        @test gradient(sum, backend, A) == ones(3, 4)
        @test gradient(a -> sum(permutedims(a, (2, 1)) .* A'), backend, A) == A
        @test gradient(a -> sum(a[2:3, :]), backend, A) == [zeros(1, 4); ones(2, 4)]
        z = ComplexF64[1+2im 3-im; -1im 2]
        @test gradient(a -> sum(abs2.(a)), backend, z) ≈ 2z
        @test gradient(a -> sum(Float64.(a .> 0)), backend, A) == zeros(3, 4)
        @test gradient(a -> sum(a * 2.0f0), backend, Float32.(A)) == fill(2.0f0, 3, 4)
    end
end
