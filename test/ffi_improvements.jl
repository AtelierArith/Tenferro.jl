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

@testset "fallible typed storage construction" begin
    err = try
        x = [1.0]
        Tenferro.RustCall.result_to_exception(
            Tenferro.RustCrate.TypedStorage(Tenferro._bytes(x), Tenferro._dims(x), Int32(-1)))
        nothing
    catch e
        e
    end
    @test err isa Tenferro.RustCall.RustError
    @test occursin("unsupported element type", sprint(showerror, err))
    @test Array(TypedTensor([1.0])) == [1.0]
end

@testset "checked slice boundary" begin
    inputs = (
        reshape(collect(1.0:6.0), 2, 3),
        view(reshape(collect(1.0:12.0), 3, 4), 1:2:3, 2:3),
        PermutedDimsArray(reshape(collect(Int32(1):Int32(6)), 2, 3), (2, 1)),
        Bool[true false; false true],
        ComplexF32[1 + 2im, 3 - 4im],
        fill(2.5),
        zeros(Float64, 0, 3),
    )
    for x in inputs, (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        @test Array(TensorType(Context(), x)) == x
    end
    for x in inputs
        t = TypedTensor(x)
        @test Array(t) == x
        isempty(x) && continue
        t[1] = x[end]
        @test t[1] == x[end]
    end

    ctx = EagerContext()
    err = try
        Tenferro.RustCall.result_to_exception(Tenferro._rust_call(ctx.graph, Tenferro.RustCrate.input_array, Tenferro._bytes([1.0, 2.0]), UInt[3], Int32(1)))
        nothing
    catch e
        e
    end
    @test err isa Tenferro.RustCall.RustError
    @test occursin("does not match the tensor shape", sprint(showerror, err))
    x = EagerTensor(ctx, ones(2, 2))
    err = try
        Tenferro.RustCall.result_to_exception(Tenferro._rust_call(ctx.graph, Tenferro.RustCrate.slice_node, UInt(x.id), UInt[0, 2]))
        nothing
    catch e
        e
    end
    @test err isa Tenferro.RustCall.RustError
    @test occursin("slice bounds", sprint(showerror, err))
end
