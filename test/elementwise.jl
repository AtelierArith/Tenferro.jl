using Test, Tenferro, DifferentiationInterface

@testset "elementwise operations and AD" begin
    for backend in (AutoTenferro(), AutoTenferroTraced()), T in (Float32, Float64)
        x = T[0.2 0.5; 0.8 1.1]
        @test gradient(a -> sum(a .* a), backend, x) ≈ 2 .* x
        @test gradient(a -> sum(exp.(a)), backend, x) ≈ exp.(x)
        @test gradient(a -> sum(sin.(a) .+ cos.(a)), backend, x) ≈ cos.(x) .- sin.(x)
        @test gradient(a -> sum(log.(a) .+ sqrt.(a)), backend, x) ≈ 1 ./ x .+ 1 ./ (2 .* sqrt.(x))
        @test gradient(a -> sum(tanh.(a)), backend, x) ≈ 1 .- tanh.(x).^2
        @test gradient(a -> sum(-a), backend, x) ≈ -ones(T, size(x))
    end
end

@testset "scalar broadcasts and shape operations" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [0.5 1.0; 1.5 2.0]
        @test gradient(a -> sum((a .+ 2) .* (3 .- a)), backend, x) ≈ 1 .- 2 .* x
        @test gradient(a -> sum(a .^ 2), backend, x) ≈ 2 .* x
        @test gradient(a -> sum(reshape(a, (4,)) .* reshape(a, (4,))), backend, x) ≈ 2 .* x
        @test gradient(a -> sum(permutedims(a, (2, 1)) * a), backend, x) ≈ 2 .* (x * ones(2,2))
        @test gradient(a -> sum(sum(a; dims=1)), backend, x) ≈ ones(2,2)
    end
end

@testset "reductions and axes" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [1.0 2.0; 3.0 4.0]
        @test gradient(a -> prod(a), backend, x) ≈ prod(x) ./ x
        @test gradient(a -> sum(prod(a; dims=1)), backend, x) ≈ [3.0 4.0; 1.0 2.0]
        @test gradient(a -> sum(maximum(a; dims=1)), backend, x) ≈ [0.0 0.0; 1.0 1.0]
        @test gradient(a -> sum(minimum(a; dims=2)), backend, x) ≈ [1.0 0.0; 1.0 0.0]
        @test gradient(a -> sum(reverse(a; dims=1) .* a), backend, x) ≈ 2 .* reverse(x; dims=1)
    end
end

@testset "mean and broadcasts" begin
    using Statistics
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [1.0 2.0; 3.0 4.0]
        @test gradient(mean, backend, x) ≈ fill(0.25, 2, 2)
        @test gradient(a -> sum(mean(a; dims=2)), backend, x) ≈ fill(0.5, 2, 2)
        @test gradient(a -> sum(a ./ 2), backend, x) ≈ fill(0.5, 2, 2)
    end
end

@testset "broadcast dimension expansion" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a, b = [1.0, 2.0], [1.0 2.0 3.0]
        da, db = gradient(z -> sum(z[1] .* z[2] .+ z[2]), backend, (a, b))
        @test da ≈ fill(sum(b), 2)
        @test db ≈ fill(sum(a) + length(a), 1, 3)
    end
end

@testset "scalar arithmetic and absolute values" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [1.0 -2.0; 3.0 -4.0]
        @test gradient(a -> sum(2 * a), backend, x) ≈ fill(2.0, 2, 2)
        @test gradient(a -> sum(a / 2), backend, x) ≈ fill(0.5, 2, 2)
        @test gradient(a -> sum(abs.(a)), backend, x) ≈ sign.(x)
    end
end

@testset "slices" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = reshape(collect(1.0:12.0), 3, 4)
        expected = zeros(3, 4); expected[1:2, 2:3] .= 1
        @test gradient(a -> sum(a[1:2, 2:3]), backend, x) == expected
        @test gradient(a -> sum(a[:, 2]), backend, x) == [0.0 1 0 0; 0 1 0 0; 0 1 0 0]
    end
end

@testset "matrix structure" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [1.0 2.0; 3.0 4.0]
        @test gradient(a -> sum(tril(a)), backend, x) == [1.0 0; 1 1]
        @test gradient(a -> sum(triu(a, 1)), backend, x) == [0.0 1; 0 0]
        @test gradient(a -> sum(diag(a)), backend, x) == [1.0 0; 0 1]
    end
end

@testset "linear solves and matrix functions" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = [3.0 1.0; 1.0 2.0]
        b = [2.0 1.0; 4.0 3.0]
        @test gradient(x -> det(x), backend, a) ≈ det(a) .* inv(a)'
        @test gradient(x -> sum(inv(x)), backend, a) ≈ -inv(a)' * ones(2,2) * inv(a)'
        da, db = gradient(z -> sum(z[1] \ z[2]), backend, (a, b))
        @test db ≈ a' \ ones(2,2)
        @test da ≈ -(a' \ ones(2,2)) * (a \ b)'
    end
end

@testset "matrix factorizations" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [3.0 0.0; 0.0 2.0]
        @test gradient(a -> sum(svd(a).S), backend, x) ≈ Matrix{Float64}(I, 2, 2)
        @test gradient(a -> sum(qr(a).R), backend, x) isa Matrix{Float64}
    end
end

@testset "pushforward and pullback of new operations" begin
    x = [0.4 0.7; 1.1 1.5]
    d = [0.2 -0.1; 0.4 0.3]
    w = [1.0 2.0; -1.0 0.5]
    f(a) = exp.(a .* a)
    (jvp,) = pushforward(f, AutoTenferro(), x, (d,))
    (vjp,) = pullback(f, AutoTenferro(), x, (w,))
    @test jvp ≈ 2 .* x .* exp.(x .* x) .* d
    @test vjp ≈ 2 .* x .* exp.(x .* x) .* w
    @test sum(jvp .* w) ≈ sum(d .* vjp)
end

@testset "concatenate tensors" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a, b = [1.0 2; 3 4], [2.0 1; 4 3]
        da, db = gradient(z -> sum(cat(z[1], z[2]; dims=2) .* cat(z[1], z[2]; dims=2)), backend, (a, b))
        @test da ≈ 2 .* a
        @test db ≈ 2 .* b
    end
end

@testset "complex magnitude and clamp" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        z = ComplexF64[3+4im, 5-12im]
        @test gradient(a -> sum(abs.(a)), backend, z) ≈ z ./ abs.(z)
        x = [-2.0, 0.2, 3.0]
        @test gradient(a -> sum(clamp.(a, -1.0, 1.0)), backend, x) ≈ [0.0, 1.0, 0.0]
    end
end

@testset "operation contracts" begin
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        ctx = Context(); a = TensorType(ctx, reshape(collect(1.0:6.0), 2, 3))
        @test Array(reshape(a, (3, 2))) == reshape(collect(1.0:6.0), 3, 2)
        @test Array(permutedims(a, (2, 1))) == permutedims(reshape(collect(1.0:6.0), 2, 3))
        @test Array(sum(a; dims=1)) == sum(reshape(collect(1.0:6.0), 2, 3); dims=1)
        @test_throws DimensionMismatch reshape(a, (4, 2))
        @test_throws BoundsError a[1:3, :]
        @test_throws DimensionMismatch a .+ TensorType(ctx, zeros(3, 2))
    end
end

@testset "Cholesky factor" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [4.0 0.0; 0.0 9.0]
        @test gradient(a -> sum(cholesky(a).L), backend, x) ≈ [0.25 0.0; 0.5 1/6]
    end
end

@testset "pseudoinverse" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = [3.0 1.0; 1.0 2.0]
        @test gradient(x -> sum(pinv(x)), backend, a) ≈ -inv(a)' * ones(2,2) * inv(a)'
    end
end

@testset "eigenvalues" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = [3.0 0.0; 0.0 2.0]
        @test gradient(x -> sum(eigen(x).values), backend, a) ≈ Matrix{Float64}(I, 2, 2)
    end
end

@testset "norm" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = ComplexF64[3+4im, 1-2im]
        @test gradient(norm, backend, a) ≈ a ./ norm(a)
    end
end

@testset "Hermitian eigendecomposition" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = [3.0 0.2; 0.2 2.0]
        @test gradient(x -> sum(eigen(Hermitian(x)).values), backend, a) ≈ Matrix{Float64}(I, 2, 2)
    end
end

@testset "gather repeated indices" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [10.0, 20.0, 30.0]
        @test gradient(a -> sum(a[[3, 1, 3]]), backend, x) == [1.0, 0.0, 2.0]
    end
end

@testset "LU factors" begin
    using LinearAlgebra, ADTypes, FiniteDiff
    a = [4.0 1.0; 2.0 3.0]
    reference = gradient(x -> sum(lu(x).U), AutoFiniteDiff(), a)
    for backend in (AutoTenferro(), AutoTenferroTraced())
        @test gradient(x -> sum(lu(x).U), backend, a) ≈ reference rtol=1e-5
    end
end

@testset "complex Hermitian eigendecomposition" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = ComplexF64[3 1+im; 1-im 4]
        @test gradient(x -> sum(eigen(Hermitian(x)).values), backend, a) ≈ Matrix{ComplexF64}(I, 2, 2)
    end
end

@testset "vector and rectangular solves" begin
    using LinearAlgebra, ADTypes, FiniteDiff
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = [3.0 0.0; 0.0 2.0]; b = [1.0, 2.0]
        ga, gb = gradient(z -> sum(z[1] \ z[2]), backend, (a,b))
        @test gb ≈ a' \ ones(2)
        @test ga ≈ -(a' \ ones(2)) * (a \ b)'
        tall = [2.0 0; 0 3; 1 1]; rhs = [1.0, 2.0, 3.0]
        expected = gradient(z -> sum(z \ rhs), AutoFiniteDiff(), tall)
        @test gradient(z -> sum(z \ rhs), backend, tall) ≈ expected rtol=1e-5
    end
end

@testset "constant Julia arrays in broadcast" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [1.0 2.0; 3.0 4.0]
        bias = [1.0 2.0]
        @test gradient(a -> sum(a .* bias), backend, x) ≈ repeat(bias, 2, 1)
        @test gradient(a -> sum(bias .- a), backend, x) ≈ -ones(2, 2)
    end
end

@testset "complex matrix factorizations" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = ComplexF64[3 0; 0 2im]
        @test gradient(a -> sum(svd(a).S), backend, x) ≈ ComplexF64[1 0; 0 im]
    end
end

@testset "strided slices" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = collect(1.0:6.0)
        @test gradient(a -> sum(a[1:2:5]), backend, x) == [1.0, 0, 1, 0, 1, 0]
    end
end

@testset "factorization reconstruction" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [3.0 0.4; 0.1 2.0]
        f(a) = (F = qr(a); sum(F.Q * F.R))
        g(a) = (F = svd(a); sum((F.U .* reshape(F.S, (1, 2))) * F.Vt))
        @test gradient(f, backend, x) ≈ ones(2,2) atol=1e-8
        @test gradient(g, backend, x) ≈ ones(2,2) atol=1e-8
    end
end

@testset "constant Julia matrices in products" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [1.0 2; 3 4]; b = [2.0 1; 4 3]
        @test gradient(a -> sum(a * b), backend, x) ≈ ones(2,2) * b'
        @test gradient(a -> sum(b * a), backend, x) ≈ b' * ones(2,2)
    end
end

@testset "remainder derivatives" begin
    a = [3.5, -3.5, 0.5]; b = [2.0, 2.0, 2.0]
    da = [1.0, 1.0, 1.0]; db = [-1.0, 1.0, 0.0]
    for backend in (AutoTenferro(), AutoTenferroTraced())
        ga, gb = gradient(z -> sum(rem.(z[1], z[2])), backend, (a, b))
        @test ga ≈ da
        @test gb ≈ db
    end
    dx, dy = [0.3, 0.2, 0.1], [0.1, 0.2, 0.3]
    (jvp,) = pushforward(z -> rem.(z[1], z[2]), AutoTenferro(), (a, b), ((dx, dy),))
    @test jvp ≈ dx .+ db .* dy
end

@testset "reduction edge cases" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        @test gradient(prod, backend, [0.0, 2, 3]) ≈ [6.0, 0, 0]
        @test gradient(prod, backend, [0.0, 0, 3]) ≈ [0.0, 0, 0]
        @test gradient(minimum, backend, [1.0, 1, 2]) ≈ [0.5, 0.5, 0]
    end
end

@testset "factorization result properties" begin
    using LinearAlgebra
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        a = [3.0 0.2; 0.2 2.0]; x = TensorType(Context(), a)
        F = svd(x); U, S, V = F
        @test Array(U) * Diagonal(Array(S)) * Array(V)' ≈ a
        @test Array(F.V) ≈ Array(V)
        Q, R = qr(x)
        @test Array(Q) * Array(R) ≈ a
        C = cholesky(x)
        @test Array(C.U) ≈ Array(C.L)'
        vals, vecs = eigen(Hermitian(x))
        @test Array(vecs) * Diagonal(Array(vals)) * Array(vecs)' ≈ a
    end
end

@testset "vector transposition" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = [1.0, 2.0, 3.0]
        @test gradient(x -> sum(transpose(x) * reshape(x, (3,1))), backend, a) ≈ 2 .* a
        z = ComplexF64[1+im, 2-im]
        @test gradient(x -> sum(adjoint(x) * reshape(x, (2,1))), backend, z) ≈ 2 .* z
    end
end

@testset "default permutations" begin
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        ctx = Context()
        for a in ([1.0, 2, 3], reshape(collect(1.0:6), 2, 3))
            @test Array(permutedims(TensorType(ctx, a))) == permutedims(a)
        end
    end
end

@testset "scalar reduction composition" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [1.0, 2.0, 3.0]
        @test gradient(a -> sum(a)^2, backend, x) ≈ fill(2sum(x), 3)
        @test gradient(a -> exp(sum(a)), backend, x) ≈ fill(exp(sum(x)), 3)
        @test gradient(a -> sum(a) + sum(a), backend, x) ≈ fill(2.0, 3)
    end
end

@testset "inferred reshape dimension" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = reshape(collect(1.0:6), 2, 3)
        @test gradient(a -> sum(reshape(a, :, 2)), backend, x) ≈ ones(2, 3)
    end
end

@testset "offset diagonals" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = reshape(collect(1.0:12.0), 3, 4)
        expected = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
        @test gradient(a -> sum(diag(a, 1)), backend, x) == expected
        expected = [0.0 0 0 0; 1 0 0 0; 0 1 0 0]
        @test gradient(a -> sum(diag(a, -1)), backend, x) == expected
    end
end

@testset "LU permutation reconstruction" begin
    using LinearAlgebra
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        a = [0.0 2.0; 1.0 3.0]
        F = lu(TensorType(Context(), a))
        @test Array(F.P) * a ≈ Array(F.L) * Array(F.U)
    end
end

@testset "triangular linear systems" begin
    using LinearAlgebra, ADTypes, FiniteDiff
    a = [3.0 8.0; 1.0 2.0]; b = [1.0, 2.0]
    f(x) = sum(LowerTriangular(x) \ b)
    reference = gradient(f, AutoFiniteDiff(), a)
    g(x) = sum(UpperTriangular(x) \ b)
    upper_reference = gradient(g, AutoFiniteDiff(), a)
    for backend in (AutoTenferro(), AutoTenferroTraced())
        @test gradient(f, backend, a) ≈ reference rtol=1e-5
        @test gradient(g, backend, a) ≈ upper_reference rtol=1e-5
    end
end

@testset "complex elementwise pushforward" begin
    for T in (ComplexF32, ComplexF64)
        x = T[0.2+0.3im, 0.7-0.2im]
        d = T[0.3-0.1im, 0.2+0.1im]
        w = T[0.4+0.2im, -0.3im]
        f(a) = exp.(a .* a)
        (df,) = pushforward(f, AutoTenferro(), x, (d,))
        (back,) = pullback(f, AutoTenferro(), x, (w,))
        @test df ≈ 2 .* x .* exp.(x .* x) .* d
        @test back ≈ conj.(2 .* x .* exp.(x .* x)) .* w
        @test real(sum(conj.(w) .* df)) ≈ real(sum(conj.(back) .* d))
    end
end

@testset "numeric promotion in broadcast" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = Float32[1, 2]
        g = gradient(a -> sum(a .+ 2.0), backend, x)
        @test g isa Vector{Float32}
        @test g ≈ ones(Float32, 2)
        a, b = Float32[1, 2], Float64[3, 4]
        da, db = gradient(z -> sum(z[1] .* z[2]), backend, (a,b))
        @test da isa Vector{Float32} && db isa Vector{Float64}
        @test da ≈ Float32.(b) && db ≈ Float64.(a)
    end
end

@testset "real and complex mixed broadcast" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = Float32[1, 2]; b = ComplexF64[2+im, 3-im]
        ga, gb = gradient(z -> sum(abs.(z[1] .* z[2])), backend, (a,b))
        @test ga ≈ Float32.(abs.(b))
        @test gb ≈ Float64.(a) .* b ./ abs.(b)
    end
end

@testset "promoted array addition" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a, b = Float32[1, 2], Float64[3, 4]
        da, db = gradient(z -> sum(z[1] + z[2]), backend, (a, b))
        @test da == ones(Float32, 2)
        @test db == ones(Float64, 2)
    end
end

@testset "promoted matrix multiplication" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = Float32[1 2; 3 4]; b = Float64[2 1; 4 3]
        da, db = gradient(z -> sum(z[1] * z[2]), backend, (a, b))
        @test da ≈ Float32.(ones(2,2) * b')
        @test db ≈ Float64.(a)' * ones(2,2)
    end
end

@testset "matrix-vector products" begin
    using LinearAlgebra
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = [1.0 2 3; 4 5 6]; x = [0.5, 1.0, 1.5]
        ga, gx = gradient(z -> sum(z[1] * z[2]), backend, (a, x))
        @test ga ≈ ones(2) * x'
        @test gx ≈ a' * ones(2)
        @test gradient(v -> sum(a * v), backend, x) ≈ a' * ones(2)
    end
end

@testset "constant Julia arrays in addition" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = Float32[1 2; 3 4]; b = Float64[1 2; 3 4]
        @test gradient(a -> sum(a + b), backend, x) == ones(Float32, 2, 2)
        @test gradient(a -> sum(b - a), backend, x) == -ones(Float32, 2, 2)
    end
end

@testset "promoted Julia matrix constants" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = Float32[1 2; 3 4]; b = Float64[2 1; 4 3]
        @test gradient(a -> sum(a * b), backend, x) ≈ Float32.(ones(2,2) * b')
        @test gradient(a -> sum(b * a), backend, x) ≈ Float32.(b' * ones(2,2))
    end
end

@testset "reshape transposed tensor" begin
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        a = reshape(collect(1.0:6), 2, 3)
        x = TensorType(Context(), a)
        @test Array(reshape(permutedims(x, (2, 1)), (6,))) == reshape(permutedims(a, (2, 1)), (6,))
    end
end

@testset "logical vector indexing" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [1.0, 2.0, 3.0]
        @test gradient(a -> sum(a[Bool[true, false, true]]), backend, x) == [1.0, 0, 1]
    end
end

@testset "scalar indexing participates in AD" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x = [1.0 2.0; 3.0 4.0]
        @test gradient(a -> a[2, 1]^2, backend, x) == [0.0 0.0; 6.0 0.0]
        @test gradient(a -> sum(a[1, 2]), backend, x) == [0.0 1.0; 0.0 0.0]
    end
end

@testset "concatenation with mixed element types" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        a = Float32[1, 2]; b = Float64[3, 4]
        da, db = gradient(z -> sum(cat(z[1], z[2]; dims=1)), backend, (a,b))
        @test da == ones(Float32, 2)
        @test db == ones(Float64, 2)
    end
end

@testset "remainder forward computation" begin
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        x = [-3.5, -0.5, 0.0, 2.5]
        a = TensorType(Context(), x)
        @test Array(rem.(a, 2.0)) == rem.(x, 2.0)
    end
end



@testset "tensor shape introspection" begin
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        x = TensorType(Context(), zeros(2,3,4))
        @test size(x) == (2,3,4)
        @test size(x, 2) == 3 && size(x, 4) == 1
        @test ndims(x) == 3 && length(x) == 24
        @test axes(x) == (Base.OneTo(2), Base.OneTo(3), Base.OneTo(4))
    end
end

@testset "eigenvalue ordering" begin
    using LinearAlgebra
    a = [0.0 -1; 1 0]
    expected = eigen(a).values
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        result = eigen(TensorType(Context(), a))
        values, vectors = Array(result.values), Array(result.vectors)
        @test values ≈ expected
        @test a * vectors ≈ vectors * Diagonal(values)
    end
end

@testset "ordered eigenvalue differentiation" begin
    using LinearAlgebra
    a = [3.0 0; 0 2.0]
    for backend in (AutoTenferro(), AutoTenferroTraced())
        @test gradient(x -> eigen(x).values[1], backend, a) ≈ [0.0 0; 0 1]
        @test gradient(x -> eigen(Hermitian(x)).values[1], backend, a) ≈ [0.0 0; 0 1]
    end
end

@testset "rectangular factorization shapes" begin
    using LinearAlgebra
    for A in ([1.0 0; 0 2; 1 1], [1.0 0 2; 0 2 1]), (Context, TensorType) in ((EagerContext,EagerTensor), (TracedContext,TracedTensor))
        x = TensorType(Context(), A)
        F = svd(x)
        @test Array(F.U) * Diagonal(Array(F.S)) * Array(F.Vt) ≈ A
        G = qr(x)
        @test Array(G.Q) * Array(G.R) ≈ A
        H = lu(x)
        @test Array(H.P) * A ≈ Array(H.L) * Array(H.U)
    end
end

@testset "ordered eigenvalue pushforward" begin
    using LinearAlgebra
    a = [3.0 0; 0 2.0]; delta = [0.2 0.0; 0.0 0.7]
    f(x) = eigen(Hermitian(x)).values[1]
    (jvp,) = pushforward(f, AutoTenferro(), a, (delta,))
    @test jvp ≈ 0.7
    (vjp,) = pullback(f, AutoTenferro(), a, (1.0,))
    @test vjp ≈ [0.0 0.0; 0.0 1.0]
end

@testset "remainder signs, broadcast and finite differences" begin
    using ADTypes, FiniteDiff
    for T in (Float32, Float64)
        a = T[3.5 -3.5 0.5; 1.25 -2.25 4.5]
        b = T[2.0 -2.0 1.0]
        f(z) = sum(rem.(z[1], z[2]))
        for backend in (AutoTenferro(), AutoTenferroTraced())
            ga, gb = gradient(f, backend, (a, b))
            @test ga ≈ ones(T, size(a))
            @test gb ≈ -sum(trunc.(a ./ b); dims=1)
        end
        direction = (fill(T(0.1), size(a)), fill(T(0.2), size(b)))
        (jvp,) = pushforward(z -> rem.(z[1], z[2]), AutoTenferro(), (a,b), (direction,))
        @test jvp ≈ direction[1] .- trunc.(a ./ b) .* direction[2]
    end
    a, b = [3.5, -3.5], [2.1, -2.3]
    g(z) = sum(rem.(z[1], z[2]))
    reference = (gradient(x -> sum(rem.(x, b)), AutoFiniteDiff(), a), gradient(y -> sum(rem.(a, y)), AutoFiniteDiff(), b))
    got = gradient(g, AutoTenferro(), (a, b))
    @test got[1] ≈ reference[1] rtol=1e-5
    @test got[2] ≈ reference[2] rtol=1e-5
    @test gradient(x -> sum(fetch(Threads.@spawn rem.(x, 2.0))), AutoTenferro(), [0.5, 1.5]) ≈ ones(2)
end

@testset "remainder of transposed tensors" begin
    x = [1.5 3.5; -2.5 4.5]
    for backend in (AutoTenferro(), AutoTenferroTraced())
        @test gradient(a -> sum(rem.(permutedims(a, (2,1)), 2.0)), backend, x) ≈ ones(2,2)
    end
end

@testset "remainder floating-point quotient" begin
    for backend in (AutoTenferro(), AutoTenferroTraced())
        x, y = [3.0, -3.0], [0.1, 0.1]
        _, dy = gradient(z -> sum(rem.(z[1], z[2])), backend, (x, y))
        @test dy ≈ -div.(x, y)
    end
end

@testset "linear algebra failure modes" begin
    using LinearAlgebra
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
        x = TensorType(Context(), [1.0 2.0; 2.0 4.0])
        @test_throws Exception Array(inv(x))
        y = TensorType(Context(), [1.0 3.0; 0.0 1.0])
        @test_throws PosDefException cholesky(y)
        @test_throws ArgumentError svd(y; full=true)
    end
end

@testset "multiple remainder nodes" begin
    a = [1.5, 3.5]
    for backend in (AutoTenferro(), AutoTenferroTraced())
        @test gradient(x -> sum(rem.(x, 2.0) + rem.(x, 3.0)), backend, a) ≈ fill(2.0, 2)
    end
end

@testset "complex sign derivatives" begin
    for backend in (AutoTenferro(), AutoTenferroTraced()), T in (ComplexF32, ComplexF64)
        z = T[1+im, 2-im]
        r = abs.(z)
        expected = (imag.(z).^2 .- im .* real.(z) .* imag.(z)) ./ (r.^3)
        @test gradient(x -> sum(sign.(x)), backend, z) ≈ expected
    end
end

@testset "complex sign forward and directional derivatives" begin
    for T in (ComplexF32, ComplexF64)
        z = T[0, 1+im, 2-im]
        for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor))
            @test Array(sign.(TensorType(Context(), z))) ≈ sign.(z)
        end
        x = z[2:3]; direction = T[0.3-0.2im, -0.1+0.4im]; seed = T[0.2+0.5im, 0.3-0.1im]
        (tangent,) = pushforward(a -> sign.(a), AutoTenferro(), x, (direction,))
        (cotangent,) = pullback(a -> sign.(a), AutoTenferro(), x, (seed,))
        expected = direction ./ abs.(x) .- x .* real.(conj.(x) .* direction) ./ abs.(x).^3
        @test tangent ≈ expected
        @test real(sum(conj.(seed) .* tangent)) ≈ real(sum(conj.(cotangent) .* direction))
    end
end

@testset "broadcast clamp bounds" begin
    a = [-2.0 0.0 4.0; -1.0 1.0 5.0]
    lower = [-1.5 0.5 1.0]
    upper = [0.5 2.0 4.5]
    f(z) = sum(clamp.(z[1], z[2], z[3]))
    for backend in (AutoTenferro(), AutoTenferroTraced())
        da, dlo, dhi = gradient(f, backend, (a, lower, upper))
        @test da == [0.0 0 1; 1 1 0]
        @test dlo == [1.0 1 0]
        @test dhi == [0.0 0 1]
    end
end

@testset "unary mathematical JVP and VJP" begin
    cases = ((exp, exp), (log, x -> inv(x)), (sqrt, x -> inv(2sqrt(x))),
             (sin, cos), (cos, x -> -sin(x)), (tanh, x -> 1 - tanh(x)^2),
             (expm1, exp), (log1p, x -> inv(1+x)))
    for T in (Float32, Float64), (op, derivative) in cases
        x = T[0.3, 0.7]; dx = T[0.2, -0.1]; seed = T[1.2, -0.4]
        f(a) = op.(a)
        (dy,) = pushforward(f, AutoTenferro(), x, (dx,))
        (back,) = pullback(f, AutoTenferro(), x, (seed,))
        @test dy ≈ derivative.(x) .* dx
        @test back ≈ derivative.(x) .* seed
    end
end

@testset "empty reductions follow Julia" begin
    using Statistics
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor)),
        T in (Float32, Float64, ComplexF32, ComplexF64), shape in ((0,), (0, 3), (2, 0)),
        op in (sum, prod, maximum, minimum, mean), dims in (1, 2)
        input = zeros(T, shape)
        tensor = TensorType(Context(), input)
        expected = try
            op(input; dims=dims)
        catch err
            err
        end
        if expected isa Exception
            @test_throws typeof(expected) op(tensor; dims=dims)
        else
            @test isequal(Array(op(tensor; dims=dims)), expected)
        end
    end
    for backend in (AutoTenferro(), AutoTenferroTraced()), op in (sum, prod),
        input in (Float64[], zeros(0, 3), zeros(2, 0))
        @test gradient(op, backend, input) == input
        @test gradient(x -> sum(op(x; dims=1)), backend, input) == input
    end
    for op in (sum, prod)
        x = zeros(0, 3)
        (dy,) = pushforward(op, AutoTenferro(), x, (x,))
        (dx,) = pullback(op, AutoTenferro(), x, (1.0,))
        @test dy == 0.0
        @test dx == x
    end
end

@testset "real mathematical domain behavior" begin
    for (Context, TensorType) in ((EagerContext, EagerTensor), (TracedContext, TracedTensor)),
        T in (Float32, Float64), (op, invalid, boundary) in ((log, -1, 0), (sqrt, -1, 0), (log1p, -2, -1))
        x = T[invalid, boundary]
        result = Array(op.(TensorType(Context(), x)))
        @test isnan(result[1])
        @test_throws DomainError op(x[1])
        @test isequal(result[2], op(x[2]))
    end
end
