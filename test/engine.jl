using Test, Tenferro, DifferentiationInterface

@testset "public names" begin
    @test Base.isexported(Tenferro, :EagerScalar)
    @test Base.ispublic(Tenferro, :DifferentiableElement)
    @test Tenferro.DifferentiableElement === Union{Float32,Float64,ComplexF32,ComplexF64}
    @test Base.ispublic(Tenferro, :num_threads) && Base.ispublic(Tenferro, :set_num_threads)
    s = sum(EagerTensor(EagerContext(), [1.0, 2.0]))
    @test s isa EagerScalar{Float64}
    @test s[] == 3.0
end

@testset "shared CPU engine threads" begin
    initial = Tenferro.num_threads()
    @test initial >= 1
    A = reshape(collect(1.0:12.0), 3, 4)
    B = reshape(collect(-6.0:5.0), 4, 3)
    ctx = EagerContext()
    a, b = EagerTensor(ctx, A), EagerTensor(ctx, B)
    serial = Array(a * b)
    try
        @test Tenferro.set_num_threads(2) == 2
        @test Tenferro.num_threads() == 2
        # Tensors created before the change stay valid and new work runs threaded.
        @test Array(a * b) ≈ serial
        @test gradient(x -> sum(x * B), AutoTenferro(), A) ≈ repeat(sum(B; dims=2)', 3)
        @test Tenferro.set_num_threads(1) == 1
        @test Array(a * b) == serial
    finally
        Tenferro.set_num_threads(initial)
    end
    @test_throws ArgumentError Tenferro.set_num_threads(0)
end
