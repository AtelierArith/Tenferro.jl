using Test, Tenferro, LinearAlgebra, DifferentiationInterface

@testset "factor shapes and reconstruction across dtypes" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        for (Context, TensorType) in
            ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            for dims in ((3, 2), (2, 3), (2, 2))
                a = reshape(T.(1:prod(dims)), dims)
                a[1, 1] += T(2)
                T <: Complex && (a[2, 1] += T(im))
                x = TensorType(Context(), a)
                m, n = dims
                k = min(m, n)
                F = svd(x)
                @test size(F.U) == (m, k) && size(F.S) == (k,) && size(F.Vt) == (k, n)
                @test Array(F.U) * Diagonal(Array(F.S)) * Array(F.Vt) ≈ a
                Q = qr(x)
                @test size(Q.Q) == (m, k) && size(Q.R) == (k, n)
                @test Array(Q.Q) * Array(Q.R) ≈ a
                F = lu(x)
                @test size(F.P) == (m, m) && size(F.L) == (m, k) && size(F.U) == (k, n)
                @test Array(F.P) * a ≈ Array(F.L) * Array(F.U)
                @test Array(pinv(x)) ≈ pinv(a) rtol=1e-4
            end
            a = T[4 1; 1 3]
            T <: Complex && (a[1, 2] += T(im); a[2, 1] -= T(im))
            ctx = Context()
            x = TensorType(ctx, a)
            F = cholesky(x)
            @test Array(F.L) * Array(F.U) ≈ a
            for uplo in (:L, :U)
                F = eigen(Hermitian(x, uplo))
                @test Array(F.vectors) * Diagonal(Array(F.values)) * Array(F.vectors)' ≈ a
            end
            F = eigen(x)
            @test a * Array(F.vectors) ≈ Array(F.vectors) * Diagonal(Array(F.values))
            @test Array(inv(x)) ≈ inv(a)
            b = T[1 2; 3 4]
            @test Array(x \ TensorType(ctx, b)) ≈ a \ b
        end
    end
end

@testset "linear algebra directional derivatives" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = T[4 0.3; 0.2 2]
        d = T[0.2 -0.1; 0.3 0.4]
        if T <: Complex
            a[1, 2] += T(0.1im)
            d[2, 1] += T(0.2im)
        end
        rhs = T[1, 2]
        objectives = (
            det,
            x -> sum(inv(x)),
            x -> sum(pinv(x)),
            norm,
            x -> sum(x \ rhs),
            x -> sum(svd(x).S),
            x -> sum(abs.(qr(x).R)),
            x -> sum(lu(x).U),
            x -> sum(eigen(x).values),
            x -> sum(eigen(Hermitian(x)).values),
            x -> sum(cholesky(Hermitian(x)).L),
            x -> sum(LowerTriangular(x) \ rhs),
            x -> sum(UpperTriangular(x) \ rhs),
        )
        h = T <: Union{Float32,ComplexF32} ? 0.002f0 : 1e-5
        for f in objectives
            reference = real(f(a + h*d) - f(a - h*d)) / (2h)
            for backend in (AutoTenferro(), AutoTenferroTraced())
                g = gradient(f, backend, a)
                @test real(sum(conj.(g) .* d)) ≈ reference rtol=0.01 atol=0.002
            end
        end
    end
end
