using Test, Tenferro

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
