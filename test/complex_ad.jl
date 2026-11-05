using Test, Tenferro, DifferentiationInterface

@testset "complex tensor autodiff" begin
    for T in (ComplexF32, ComplexF64)
        a = T[1+2im 2-im; -1+im 3+2im]
        b = T[2-im 1+2im; -3im 1-im]
        w = T[1+2im 2-im; 1-im -2+3im]
        for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            ctx = Context()
            x, y = TensorType(ctx, a), TensorType(ctx, b)
            @test Array(x) == a
            @test Array(x+y) ≈ a+b
            @test Array(x*y) ≈ a*b
        end
        for backend in (AutoTenferro(), AutoTenferroTraced())
            da, db = gradient(z -> sum(z[1] * z[2]), backend, (a, b))
            @test da isa Matrix{T}
            @test da ≈ ones(T, 2, 2) * b'
            @test db ≈ a' * ones(T, 2, 2)
            @test gradient(x -> sum(x + x), backend, T[1+im, 2-im]) == fill(T(2), 2)
        end
        ((da, db),) = pullback(z -> z[1] * z[2], AutoTenferro(), (a,b), (w,))
        @test da ≈ w * b'
        @test db ≈ a' * w
    end
end

@testset "tenferro built-in non-differentiable types" begin
    for a in (Int32[1 2; -3 4], Int64[1 2; -3 4], Bool[1 0; 0 1])
        t = TypedTensor(a)
        @test Array(t) == a
        t[1] = a[2]
        @test t[1] == a[2]
        for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            ctx = Context()
            x = TensorType(ctx, a)
            @test Array(x) == a
            if eltype(a) != Bool
                @test Array(x + x) == a + a
            end
        end
    end
end

@testset "complex transpose and adjoint" begin
    for T in (ComplexF32, ComplexF64)
        a = T[1+2im 3-im; -2+im 4+3im]
        seed = T[1+im 2-im; -3+2im 4-im]
        for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            x = TensorType(Context(), a)
            @test Array(transpose(x)) == transpose(a)
            @test Array(adjoint(x)) == adjoint(a)
        end
        (dt,) = pullback(transpose, AutoTenferro(), a, (seed,))
        (da,) = pullback(adjoint, AutoTenferro(), a, (seed,))
        @test dt ≈ transpose(seed)
        @test da ≈ adjoint(seed)
        for backend in (AutoTenferro(), AutoTenferroTraced())
            @test gradient(x -> sum(transpose(x)), backend, a) ≈ ones(T, size(a))
            @test gradient(x -> sum(adjoint(x)), backend, a) ≈ ones(T, size(a))
        end
    end
end

@testset "tensor input and output ownership" begin
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        parent = reshape(collect(1.0:12.0), 3, 4)
        source = @view parent[:, 2:3]
        expected = copy(source)
        ctx = Context()
        x = TensorType(ctx, source)
        source .= 0
        GC.gc(true)
        @test Array(x) == expected
        result = Array(x)
        result .= -1
        GC.gc(true)
        @test Array(x) == expected
        y = reshape(permutedims(x, (2, 1)), (6,))
        x = nothing
        ctx = nothing
        GC.gc(true)
        @test Array(y) == reshape(permutedims(expected, (2, 1)), (6,))
    end
end

@testset "complex elementary functions" begin
    cases = ((exp, exp), (log, inv), (sqrt, z -> inv(2sqrt(z))),
             (sin, cos), (cos, z -> -sin(z)), (tanh, z -> 1 - tanh(z)^2),
             (expm1, exp), (log1p, z -> inv(1 + z)))
    for T in (ComplexF32, ComplexF64), (op, derivative) in cases
        x = T[0.4+0.2im, 0.8-0.3im]
        direction = T[0.2-0.1im, -0.3+0.4im]
        seed = T[0.3+0.5im, -0.2+0.1im]
        for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            @test Array(op.(TensorType(Context(), x))) ≈ op.(x)
        end
        (dy,) = pushforward(a -> op.(a), AutoTenferro(), x, (direction,))
        (back,) = pullback(a -> op.(a), AutoTenferro(), x, (seed,))
        @test dy ≈ derivative.(x) .* direction
        @test back ≈ conj.(derivative.(x)) .* seed
        for backend in (AutoTenferro(), AutoTenferroTraced())
            @test gradient(a -> sum(op.(a)), backend, x) ≈ conj.(derivative.(x))
        end
    end
end

@testset "complex reductions" begin
    using Statistics
    for T in (ComplexF32,ComplexF64)
        a = T[1+im 2-im; 3-im 4+im]
        for (C,S) in ((EagerContext,EagerTensor),(TracedContext,TracedTensor))
            for op in (sum,prod,mean), dims in (1,2,(1,2))
                @test Array(op(S(C(),a);dims=dims)) ≈ op(a;dims=dims)
            end
        end
        for backend in (AutoTenferro(),AutoTenferroTraced())
            @test gradient(sum,backend,a) ≈ ones(T,size(a))
            @test gradient(mean,backend,a) ≈ fill(T(1/4),size(a))
            @test gradient(prod,backend,a) ≈ conj.(prod(a)./a)
        end
    end
end
