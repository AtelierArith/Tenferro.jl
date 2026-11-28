using Test, Tenferro, NNlib
using DifferentiationInterface, FiniteDiff
import ChainRulesCore
const DI = DifferentiationInterface

const BACKENDS = (AutoTenferro(), AutoTenferroTraced())
const CONTEXTS = (EagerContext, TracedContext)
tensor(ctx, a) = ctx isa EagerContext ? EagerTensor(ctx, a) : TracedTensor(ctx, a)

# Values of `f` on tensors (eager and traced) agree with NNlib on plain arrays.
function check_values(f, args...; rtol=1e-5)
    reference = f(args...)
    for C in CONTEXTS
        ctx = C()
        result = Array(f(map(a -> tensor(ctx, a), args)...))
        size(result) == size(reference) && isapprox(result, reference; rtol=rtol) || return false
    end
    true
end

# Gradients of `sum(f(args...; kw...) .* weights)` with respect to `args[wrt]`
# match NNlib's own ChainRules pullback; the other arguments stay plain arrays.
function nnlib_gradient(f, args...; wrt=1, kw...)
    y, pullback = ChainRulesCore.rrule(f, args...; kw...)
    weights = reshape(collect(range(-1, 1; length=length(y))), size(y))
    weights, ChainRulesCore.unthunk(pullback(weights)[wrt + 1])
end
function check_gradient(f, args...; wrt=1, rtol=1e-6, kw...)
    weights, expected = nnlib_gradient(f, args...; wrt=wrt, kw...)
    with(a) = ntuple(i -> i == wrt ? a : args[i], length(args))
    all(BACKENDS) do backend
        value, grad = value_and_gradient(a -> sum(f(with(a)...; kw...) .* weights), backend, args[wrt])
        value ≈ sum(f(args...; kw...) .* weights) && isapprox(grad, expected; rtol=rtol)
    end
end

const CONV_CONFIGS = (
    (stride=1, pad=0, dilation=1, groups=1, flipped=false),
    (stride=2, pad=1, dilation=1, groups=1, flipped=true),
    (stride=(1, 2, 1), pad=:asym, dilation=2, groups=2, flipped=false),
    (stride=2, pad=1, dilation=1, groups=4, flipped=true),
)
spatial(c, S) = c isa Tuple ? c[1:S] : c
padding(c, S) = c === :asym ? ntuple(i -> isodd(i) ? 2 : 1, 2S) : c

@testset "conv, ∇conv_data and ∇conv_filter match NNlib in 1-3 spatial dimensions" begin
    for S in 1:3, c in CONV_CONFIGS
        x = randn(ntuple(d -> 6 + d, S)..., 4, 2)
        w = randn(ntuple(d -> d + 1, S)..., 4 ÷ c.groups, 4)
        kw = (stride=spatial(c.stride, S), pad=padding(c.pad, S), dilation=c.dilation, groups=c.groups, flipped=c.flipped)
        cdims = DenseConvDims(size(x), size(w); stride=kw.stride, padding=kw.pad, dilation=kw.dilation,
                              groups=kw.groups, flipkernel=kw.flipped)
        dy = randn(NNlib.output_size(cdims)..., 4, 2)
        @test check_values((x, w) -> conv(x, w; kw...), x, w)
        @test check_values((x, w) -> conv(x, w, cdims), x, w)
        @test check_values((dy, w) -> ∇conv_data(dy, w, cdims), dy, w)
        @test check_values((x, dy) -> ∇conv_filter(x, dy, cdims), x, dy)
        # Gradients with respect to the input and to the kernel (the other operand
        # stays a plain array, exercising the mixed methods).
        @test check_gradient(conv, x, w, cdims)
        @test check_gradient(conv, x, w, cdims; wrt=2)
        @test check_gradient(∇conv_data, dy, w, cdims)
    end
end

@testset "conv gradients agree with finite differences" begin
    x, w = randn(6, 5, 2, 2), randn(3, 2, 2, 3)
    weights = randn(5, 4, 3, 2)
    for backend in BACKENDS
        lossx(a) = sum(conv(a, w; stride=(1, 2), pad=(1, 0, 2, 1)) .* weights)
        lossw(b) = sum(conv(x, b; stride=(1, 2), pad=(1, 0, 2, 1)) .* weights)
        @test gradient(lossx, backend, x) ≈ FiniteDiff.finite_difference_gradient(lossx, x) rtol = 1e-6
        @test gradient(lossw, backend, w) ≈ FiniteDiff.finite_difference_gradient(lossw, w) rtol = 1e-6
    end
end

@testset "depthwise convolution" begin
    for S in 1:3
        x = randn(ntuple(_ -> 6, S)..., 3, 2)
        w = randn(ntuple(_ -> 3, S)..., 2, 3)
        cdims = DepthwiseConvDims(size(x), size(w); stride=2, padding=1)
        dy = randn(NNlib.output_size(cdims)..., 6, 2)
        @test check_values((x, w) -> depthwiseconv(x, w; stride=2, pad=1, flipped=true), x, w)
        @test check_values((x, w) -> depthwiseconv(x, w, cdims), x, w)
        @test check_values((dy, w) -> ∇depthwiseconv_data(dy, w, cdims), dy, w)
        @test check_values((x, dy) -> ∇depthwiseconv_filter(x, dy, cdims), x, dy)
        @test check_gradient(depthwiseconv, x, w, cdims)
        @test check_gradient(depthwiseconv, x, w, cdims; wrt=2)
    end
end

@testset "Float32 convolution and dims constructors" begin
    x, w = randn(Float32, 8, 8, 3, 2), randn(Float32, 3, 3, 3, 4)
    for C in CONTEXTS
        ctx = C()
        xt, wt = tensor(ctx, x), tensor(ctx, w)
        y = conv(xt, wt; pad=1)
        @test eltype(y) === Float32
        @test Array(y) ≈ conv(x, w; pad=1) rtol = 1e-5
        @test DenseConvDims(xt, wt; stride=2) == DenseConvDims(x, w; stride=2)
        @test DepthwiseConvDims(xt, randn(3, 3, 1, 3)) == DepthwiseConvDims(x, randn(3, 3, 1, 3))
        @test PoolDims(xt, 2) == PoolDims(x, 2)
        @test Array(conv(x, wt)) ≈ conv(x, w) rtol = 1e-5
    end
    value, grad = value_and_gradient(a -> sum(conv(a, w; stride=2)), AutoTenferro(), x)
    @test value isa Float32 && grad isa Array{Float32,4}
end

@testset "pooling matches NNlib in 1-3 spatial dimensions" begin
    for S in 1:3
        x = randn(ntuple(_ -> 7, S)..., 3, 2)
        k = ntuple(_ -> 2, S)
        asym = ntuple(i -> isodd(i) ? 1 : 0, 2S)
        pdims = (PoolDims(size(x), k), PoolDims(size(x), k; padding=1, stride=1),
                 PoolDims(size(x), k; padding=asym, stride=2, dilation=2),
                 PoolDims(size(x), (3, k[2:end]...); padding=1, stride=2, ceil_mode=true))
        for pd in pdims
            @test check_values(x -> maxpool(x, pd), x)
            @test check_values(x -> meanpool(x, pd), x)
            @test check_values(x -> meanpool(x, pd; count_include_pad=false), x)
            @test check_values(x -> lpnormpool(x, pd; p=2), abs.(x))
            @test check_gradient(maxpool, x, pd)
            @test check_gradient(meanpool, x, pd)
            @test check_gradient(meanpool, x, pd; count_include_pad=false)
            @test check_gradient(lpnormpool, abs.(x) .+ 0.1, pd; p=3, rtol=1e-5)
        end
        @test check_values(x -> maxpool(x, k; pad=1, stride=1), x)
        @test check_values(x -> meanpool(x, k; stride=2, ceil_mode=true, count_include_pad=false), x)
        @test check_values(x -> lpnormpool(x, 1.5, k; pad=1), abs.(x))
        # Lux's LPPool default: integer p on signed inputs.
        @test check_values(x -> lpnormpool(x, 2, k; stride=1), x)
        @test check_gradient(lpnormpool, x, PoolDims(size(x), k; stride=1); p=2)
    end
end

@testset "maxpool routes each window's gradient to its first maximum" begin
    # ReLU outputs have many tied zeros; NNlib's ∇maxpool picks the first maximum
    # in kernel order instead of splitting the gradient.
    x = max.(round.(randn(6, 6, 2, 2)), 0.0)
    for pd in (PoolDims(size(x), (2, 2)), PoolDims(size(x), (3, 3); stride=1, padding=1))
        _, expected = nnlib_gradient(maxpool, x, pd)
        weights = reshape(collect(range(-1, 1; length=prod(NNlib.output_size(pd)) * 4)), NNlib.output_size(pd)..., 2, 2)
        for backend in BACKENDS
            @test gradient(a -> sum(maxpool(a, pd) .* weights), backend, x) ≈ expected rtol = 1e-12
        end
    end
end

@testset "global and adaptive pooling as Lux calls them" begin
    x = randn(7, 6, 3, 2)
    for C in CONTEXTS
        xt = tensor(C(), x)
        @test Array(maximum(xt; dims=1:2, init=-Inf)) ≈ maximum(x; dims=1:2, init=-Inf)
        @test Array(minimum(xt; dims=(1, 2), init=Inf)) ≈ minimum(x; dims=(1, 2), init=Inf)
        @test Array(lpnormpool(xt, PoolDims(xt, size(x)[1:2]); p=2)) ≈ lpnormpool(x, PoolDims(x, size(x)[1:2]); p=2)
        # AdaptiveMeanPool((2, 3)) on a 7×6 input.
        stride = (7, 6) .÷ (2, 3)
        pd = PoolDims(xt, (7, 6) .- ((2, 3) .- 1) .* stride; padding=0, stride=stride, dilation=1)
        @test Array(meanpool(xt, pd)) ≈ meanpool(x, pd)
    end
    for backend in BACKENDS
        loss(a) = sum(maximum(a; dims=1:2, init=-Inf) .* reshape(1.0:6.0, 1, 1, 3, 2))
        @test gradient(loss, backend, x) ≈ FiniteDiff.finite_difference_gradient(loss, x) rtol = 1e-6
    end
end

@testset "upsampling and pixel shuffle" begin
    x = randn(4, 5, 2, 2)
    @test check_values(x -> upsample_nearest(x, (2, 3)), x)
    @test check_values(x -> upsample_nearest(x; size=(8, 5)), x)
    @test check_values(x -> upsample_nearest(x, (2, 2, 2)), randn(3, 2, 2, 2, 1))
    @test check_gradient(upsample_nearest, x, (2, 3))
    for ac in (true, false)
        @test check_values(x -> upsample_bilinear(x, (2, 1.5); align_corners=ac), x)
        @test check_values(x -> upsample_bilinear(x; size=(3, 9), align_corners=ac), x)
        @test check_values(x -> upsample_linear(x, 2.5; align_corners=ac), randn(4, 2, 3))
        @test check_values(x -> upsample_trilinear(x, (2, 2, 3); align_corners=ac), randn(3, 4, 2, 2, 1))
        @test check_values(x -> upsample_trilinear(x; size=(5, 4, 1), align_corners=ac), randn(3, 4, 2, 2, 1))
        weights = randn(8, 7, 2, 2)
        for backend in BACKENDS
            loss(a) = sum(upsample_bilinear(a; size=(8, 7), align_corners=ac) .* weights)
            @test gradient(loss, backend, x) ≈ FiniteDiff.finite_difference_gradient(loss, x) rtol = 1e-6
        end
    end
    @test check_values(x -> pixel_shuffle(x, 2), randn(3, 4, 8, 2))
    @test check_values(x -> pixel_shuffle(x, 2), randn(3, 4, 2, 16, 2))
    @test check_values(x -> pixel_shuffle(x, 3), randn(3, 9, 2))
    for backend in BACKENDS
        weights = randn(6, 8, 2, 2)
        loss(a) = sum(pixel_shuffle(a, 2) .* weights)
        a = randn(3, 4, 8, 2)
        @test gradient(loss, backend, a) ≈ FiniteDiff.finite_difference_gradient(loss, a) rtol = 1e-6
    end
end

@testset "a small CNN trains end to end" begin
    # conv → relu → maxpool → conv-transpose → upsample, differentiated with
    # respect to every kernel at once.
    x = randn(Float32, 8, 8, 1, 2)
    params = (w1=randn(Float32, 3, 3, 1, 4) ./ 3, w2=randn(Float32, 2, 2, 2, 4) ./ 2)
    function model(p)
        h = relu.(conv(x, p.w1; pad=1))
        h = maxpool(h, (2, 2))
        cd = DenseConvDims((8, 8, 2, 2), size(p.w2); stride=2)
        h = ∇conv_data(h, p.w2, cd)
        u = upsample_bilinear(h, (2, 2))
        sum(u .* u)
    end
    for backend in BACKENDS
        value, grad = value_and_gradient(model, backend, params)
        @test value ≈ model(params) rtol = 1e-4
        @test grad.w1 isa Array{Float32,4} && size(grad.w2) == size(params.w2)
        ε = 1.0f-2
        step = map((p, g) -> p .- ε .* g ./ (sqrt(sum(abs2, g)) + 1.0f-6), params, grad)
        @test model(step) < value
    end
end
