using Test, Tenferro, NNlib, SpecialFunctions
using DifferentiationInterface, FiniteDiff
const DI = DifferentiationInterface

const BACKENDS = (AutoTenferro(), AutoTenferroTraced())
fd_gradient(f, x) = FiniteDiff.finite_difference_gradient(f, x)

# Points away from every kink, so finite differences are reliable.
const SMOOTH_POINTS = [-2.5, -1.3, -0.3, 0.4, 1.2, 3.7]
const ACTIVATIONS = (
    relu, relu6, leakyrelu, elu, celu, selu, gelu, gelu_tanh, gelu_erf, gelu_sigmoid, quick_gelu,
    swish, hardswish, sigmoid, sigmoid_fast, hardσ, logσ, hardtanh, tanh_fast, softplus, softsign,
    mish, lisht, tanhshrink, logcosh, softshrink, trelu,
)

# NNlib's ChainRules derivatives at the kinks (also what Zygote returns).
nnlib_derivative(::typeof(relu), x) = x > 0 ? 1.0 : 0.0
nnlib_derivative(::typeof(relu6), x) = 0 < x < 6 ? 1.0 : 0.0
nnlib_derivative(::typeof(leakyrelu), x) = x > 0 ? 1.0 : 0.01
nnlib_derivative(::typeof(elu), x) = x >= 0 ? 1.0 : exp(x)
nnlib_derivative(::typeof(celu), x) = x >= 0 ? 1.0 : exp(x)
nnlib_derivative(::typeof(selu), x) = x > 0 ? NNlib.selu_λ : NNlib.selu_λ * NNlib.selu_α * exp(x)
nnlib_derivative(::typeof(hardswish), x) = NNlib.deriv_hardswish(x)
nnlib_derivative(::typeof(hardσ), x) = -3 < x < 3 ? 1 / 6 : 0.0
nnlib_derivative(::typeof(hardtanh), x) = -1 < x < 1 ? 1.0 : 0.0
nnlib_derivative(::typeof(softshrink), x) = abs(x) > 0.5 ? 1.0 : 0.0
nnlib_derivative(::typeof(trelu), x) = x > 1 ? 1.0 : 0.0
nnlib_derivative(::typeof(softplus), x) = sigmoid(x)
const KINKS = [0.0, -3.0, 3.0, 6.0, 1.0, -1.0, 0.5, -0.5]

@testset "activation broadcasts match NNlib values and gradients" begin
    w = collect(1.0:length(SMOOTH_POINTS))
    for backend in BACKENDS, f in ACTIVATIONS
        loss(a) = sum(f.(a) .* w)
        value, grad = value_and_gradient(loss, backend, SMOOTH_POINTS)
        @test value ≈ loss(SMOOTH_POINTS) rtol = 1e-6
        @test grad ≈ fd_gradient(loss, SMOOTH_POINTS) rtol = 1e-5 atol = 1e-7
    end
end

@testset "activation derivatives at kinks follow NNlib's rules" begin
    for backend in BACKENDS
        for f in (relu, relu6, leakyrelu, elu, celu, selu, hardswish, hardσ, hardtanh, softshrink, trelu, softplus)
            @test gradient(a -> sum(f.(a)), backend, KINKS) ≈ nnlib_derivative.(f, KINKS)
        end
        # The unselected `expm1` branch stays finite for large inputs.
        @test all(isfinite, gradient(a -> sum(elu.(a) .+ selu.(a)), backend, [1000.0, -1000.0]))
        @test gradient(a -> sum(relu.(a)), backend, [-1.0, 0.0, 2.0]) == [0.0, 0.0, 1.0]
    end
end

@testset "activation parameters, element types and scalars" begin
    x = [-2.0, -0.4, 0.7, 1.6]
    for backend in BACKENDS
        for (f, p) in ((leakyrelu, 0.2), (elu, 0.5), (celu, 2.0), (softshrink, 0.3), (trelu, 0.5))
            loss(a) = sum(f.(a, p))
            @test gradient(loss, backend, x) ≈ fd_gradient(loss, x) rtol = 1e-5
            @test value_and_gradient(loss, backend, x)[1] ≈ loss(x)
        end
        x32 = Float32[-1, 0.5, 2]
        value, grad = value_and_gradient(a -> sum(gelu.(a) .+ relu.(a)), backend, x32)
        @test value isa Float32 && grad isa Vector{Float32}
        @test value ≈ sum(gelu.(x32) .+ relu.(x32)) rtol = 1e-4
    end
    s = sum(EagerTensor(EagerContext(), [1.0, -3.0]))
    @test relu.(s)[] == 0.0
    @test sigmoid.(s)[] ≈ sigmoid(-2.0)
    x = EagerTensor(EagerContext(), [1.0, 2.0])
    @test NNlib.fast_act(tanh, x) === tanh
    @test NNlib.fast_act(sigmoid, x) === sigmoid
end

@testset "softmax, logsoftmax and logsumexp" begin
    x = [0.3 -1.2 2.0 0.1; 1.5 0.4 -0.7 0.9; -0.2 1.1 0.6 -1.4;;; 0.8 -0.5 1.3 0.2; -1.0 0.7 0.0 2.2; 0.5 -0.3 1.9 -0.6]
    w = reshape(collect(1.0:length(x)), size(x)) ./ 7
    for backend in BACKENDS
        for f in (softmax, logsoftmax, logsumexp), dims in (1, 2, 3, (1, 2), :, 4)
            loss(a) = sum(f(a; dims=dims) .* w)
            value, grad = value_and_gradient(loss, backend, x)
            @test value ≈ loss(x)
            @test grad ≈ fd_gradient(loss, x) rtol = 1e-5 atol = 1e-8
        end
        # Default dims: softmax and logsoftmax along 1, logsumexp over everything.
        defaults(a) = softmax(a)[2, 1, 1] + logsoftmax(a)[1, 2, 2] + logsumexp(a)
        @test value_and_gradient(defaults, backend, x)[1] ≈ defaults(x)
        @test gradient(defaults, backend, x) ≈ fd_gradient(defaults, x) rtol = 1e-5
        # Large inputs stay finite thanks to the max shift.
        big = [1000.0, 1001.0, 999.0]
        @test gradient(a -> logsumexp(a) + sum(softmax(a) .* [1.0, 2.0, 3.0]), backend, big) ≈
              fd_gradient(a -> logsumexp(a) + sum(softmax(a) .* [1.0, 2.0, 3.0]), big) rtol = 1e-4
    end
end

@testset "batched_mul, batched_transpose and batched_adjoint" begin
    cases = (
        (randn(2, 3, 4), randn(3, 5, 4), (a, b) -> a ⊠ b),
        (randn(2, 3, 4), randn(5, 3, 4), (a, b) -> batched_mul(a, batched_transpose(b))),
        (randn(3, 2, 4), randn(3, 5, 4), (a, b) -> batched_adjoint(a) ⊠ b),
        (randn(2, 3, 4), randn(3, 5, 1), (a, b) -> a ⊠ b),
        (randn(2, 3, 1), randn(3, 5, 4), (a, b) -> a ⊠ b),
        (randn(2, 3), randn(3, 5, 4), (a, b) -> a ⊠ b),
        (randn(2, 3, 4), randn(3, 5), (a, b) -> a ⊠ b),
        (randn(2, 3, 2, 2), randn(3, 5, 2, 2), (a, b) -> a ⊠ b),
    )
    for backend in BACKENDS, (A, B, op) in cases
        W = randn(size(op(A, B)))
        loss(p) = sum(op(p.a, p.b) .* W)
        value, grad = value_and_gradient(loss, backend, (a=A, b=B))
        @test value ≈ loss((a=A, b=B))
        @test grad.a ≈ fd_gradient(a -> loss((a=a, b=B)), A) rtol = 1e-5 atol = 1e-8
        @test grad.b ≈ fd_gradient(b -> loss((a=A, b=b)), B) rtol = 1e-5 atol = 1e-8
    end
    for backend in BACKENDS
        A, B = randn(2, 3, 4), randn(3, 5, 4)
        @test gradient(a -> sum(batched_mul(a, B)), backend, A) ≈ fd_gradient(a -> sum(batched_mul(a, B)), A) rtol = 1e-5
        @test gradient(b -> sum(batched_mul(A, b)), backend, B) ≈ fd_gradient(b -> sum(batched_mul(A, b)), B) rtol = 1e-5
        C = ComplexF64.(randn(2, 3, 2), randn(2, 3, 2))
        ctx = backend isa AutoTenferro ? EagerContext() : TracedContext()
        t = backend isa AutoTenferro ? EagerTensor(ctx, C) : TracedTensor(ctx, C)
        @test Array(batched_adjoint(t)) ≈ conj(permutedims(C, (2, 1, 3)))
        @test Array(batched_transpose(t)) ≈ permutedims(C, (2, 1, 3))
    end
    ctx = EagerContext()
    @test_throws DimensionMismatch EagerTensor(ctx, randn(2, 3, 4)) ⊠ EagerTensor(ctx, randn(3, 5, 3))
    @test_throws DimensionMismatch EagerTensor(ctx, randn(2, 3, 4)) ⊠ EagerTensor(ctx, randn(4, 5, 4))
end

@testset "pad_zeros and pad_constant" begin
    M = [1.0 -2.0 3.0 0.5; 0.2 4.0 -1.0 2.5; -0.7 1.5 0.3 -2.0]
    cases = (
        a -> pad_zeros(a, 2),
        a -> pad_zeros(a, (1, 2, 0, 3)),
        a -> pad_zeros(a, (1, 1); dims=2),
        a -> pad_constant(a, (1, 2, 0, 3), 2.5),
        a -> pad_constant(a, ((1, 0), (0, 2)), -1.0),
        a -> pad_constant(a, 1, 3.0; dims=1),
    )
    for backend in BACKENDS, pad in cases
        W = reshape(collect(1.0:length(pad(M))), size(pad(M)))
        loss(a) = sum(abs2.(pad(a)) .* W)
        value, grad = value_and_gradient(loss, backend, M)
        @test value ≈ loss(M)
        @test grad ≈ fd_gradient(loss, M) rtol = 1e-6
    end
    ctx = EagerContext()
    t = EagerTensor(ctx, M)
    @test Array(pad_constant(t, (1, 2, 0, 3), 2.5)) == pad_constant(M, (1, 2, 0, 3), 2.5)
    # Interior (dilation) padding, used for transposed convolutions.
    @test Array(Tenferro._pad(EagerTensor(ctx, [1.0 2.0; 3.0 4.0]), (0, 1), (0, 0), (1, 2))) ==
          [0 1 0 0 2; 0 0 0 0 0; 0 3 0 0 4]
    @test_throws ArgumentError pad_zeros(t, (1, 2, 3))
end

@testset "broadcast fallback and composed functions" begin
    for backend in BACKENDS
        @test gradient(a -> sum((relu ∘ +).(a, 1.0)), backend, [-2.0, 0.5]) == [0.0, 1.0]
        @test gradient(a -> sum(identity.(a)), backend, [-2.0, 0.5]) == [1.0, 1.0]
        err = try
            gradient(a -> sum(cbrt.(a)), backend, [1.0])
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("cbrt", err.msg)
        @test_throws ArgumentError gradient(a -> sum((t -> 2t).(a)), backend, [1.0])
    end
    s = sum(EagerTensor(EagerContext(), [1.0, 2.0]))
    @test Array(s .* [1.0, 2.0]) == [3.0, 6.0]
    @test ([2.0, 4.0] ./ s isa EagerTensor)
    @test (s .+ 1)[] == 4.0
end
