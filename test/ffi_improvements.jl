using Test, Tenferro, ChainRulesCore

@testset "pullback graph reuse and aliasing" begin
    ctx = EagerContext()
    a = ComplexF64[1+2im 3-im; 2im 1+im]
    x = EagerTensor(ctx, a)
    z, back = rrule(*, x, x)
    cot = EagerTensor(ctx, ComplexF64[1+im 3im; -2 2-im])
    _, dx, dy = back(cot)
    @test Array(dx) ≈ Array(cot) * a'
    @test Array(dy) ≈ a' * Array(cot)
    @test Array(dx + dy) ≈ Array(cot) * a' + a' * Array(cot)
    _, dx2, dy2 = back(cot)
    @test Array(dx2) ≈ Array(dx)
    @test Array(dy2) ≈ Array(dy)
    other_seed = EagerTensor(EagerContext(), Array(cot))
    _, dx3, dy3 = back(other_seed)
    @test Array(dx3) ≈ Array(dx)
    @test Array(dy3) ≈ Array(dy)
end

@testset "Rust errors and context synchronization" begin
    ctx = EagerContext()
    flags = EagerTensor(ctx, Bool[true, false])
    err = try
        flags + flags
        nothing
    catch e
        e
    end
    @test !isnothing(err)
    @test occursin("Bool", sprint(showerror, err))
    x = EagerTensor(ctx, ones(2, 2))
    tasks = map(1:16) do _
        Threads.@spawn Array(x + x)
    end
    @test all(fetch(t) == fill(2.0, 2, 2) for t in tasks)
end

@testset "context lifetime" begin
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        ctx = Context()
        x = TensorType(ctx, ones(2, 2))
        @test Array(x) == ones(2, 2)
        close(ctx)
        close(ctx)
        @test_throws ArgumentError Array(x)
        @test_throws ArgumentError TensorType(ctx, ones(2, 2))
        @test Array(TensorType(Context(), ones(1, 1))) == ones(1, 1)
    end
end
