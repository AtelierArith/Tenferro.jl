using Test, Tenferro, LinearAlgebra, DifferentiationInterface

@testset "factorization reconstruction across types" begin
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor)),
        T in (Float32, Float64, ComplexF32, ComplexF64)
        base = T[2 1; 1 3; 0 1]
        if T <: Complex
            base .+= im .* T[0.3 0.2; -0.1 0.1; 0.2 -0.3]
        end
        tol = T <: Union{Float32,ComplexF32} ? 1e-5 : 1e-12
        for a in (base, copy(adjoint(base)))
            x = TensorType(Context(), a)
            f = svd(x)
            @test Array(f.U) * Diagonal(Array(f.S)) * Array(f.Vt) ≈ a rtol=tol atol=tol
            q = qr(x)
            @test Array(q.Q) * Array(q.R) ≈ a rtol=tol atol=tol
            l = lu(x)
            @test Array(l.P) * a ≈ Array(l.L) * Array(l.U) rtol=tol atol=tol
        end
        h = base' * base + 2I
        x = TensorType(Context(), Matrix(h))
        c = cholesky(Hermitian(x))
        @test Array(c.L) * Array(c.L)' ≈ h rtol=tol atol=tol
        e = eigen(Hermitian(x))
        @test Array(e.vectors) * Diagonal(Array(e.values)) * Array(e.vectors)' ≈ h rtol=tol atol=tol
    end
end

@testset "wide least squares is unsupported" begin
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        ctx = Context()
        a = TensorType(ctx, [1.0 0.0 1.0; 0.0 1.0 1.0])
        b = TensorType(ctx, [1.0, 2.0])
        @test_throws Exception a \ b
    end
end

@testset "factorization directional gradients" begin
    for T in (Float32,Float64,ComplexF32,ComplexF64)
        a = T[3 0.2; 0.1 2]
        direction = T[0.2 -0.1; 0.3 0.4]
        if T <: Complex
            a .+= im .* T[0.1 0.2; -0.1 0.3]
            direction .*= 1+0.2im
        end
        h = T <: Union{Float32,ComplexF32} ? 0.002f0 : 1e-5
        functions = (x -> sum(abs.(inv(x))), x -> sum(abs.(pinv(x))),
                     x -> sum(svd(x).S), x -> sum(abs.(qr(x).R)),
                     x -> sum(abs.(lu(x).U)),
                     x -> sum(eigen(Hermitian(x)).values),
                     x -> sum(abs.(cholesky(Hermitian(x)).L)))
        for f in functions
            numerical = (f(a+h*direction) - f(a-h*direction)) / (2h)
            for backend in (AutoTenferro(),AutoTenferroTraced())
                g = gradient(f,backend,a)
                @test real(sum(conj.(g).*direction)) ≈ numerical rtol=(h == 0.002f0 ? 0.02 : 1e-5) atol=(h == 0.002f0 ? 0.002 : 1e-7)
            end
        end
    end
end
