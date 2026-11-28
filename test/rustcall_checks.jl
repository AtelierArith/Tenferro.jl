using Test, Tenferro
const RustCall = Tenferro.RustCall

# An aggregate RustCall has no layout assertion for.
struct UnregisteredPair
    a::Int64
    b::Int64
end

@testset "by-value checks of bridge results are decided at compile time" begin
    types = Tenferro._bridge_result_types()
    # Every `Result`-returning bridge method has one; the eager graph alone has dozens.
    @test length(types) > 50
    @test Tenferro.RustCrate.CResult_EagerGraph_input_array in types
    @test all(T -> T <: RustCall.FFIByValue, types)
    plain = Tuple{Ptr{Nothing},Ptr{UInt8},UInt64,Int32}
    for T in types
        m = which(RustCall.ffi_check_by_value_signature, Tuple{Type{T},Type{plain}})
        @test m.module === Tenferro
        # The whole check compiles to `nothing`.
        @test only(Base.return_types(RustCall.ffi_check_by_value_signature, (Type{T}, Type{plain}))) === Nothing
        @test RustCall.ffi_check_by_value_signature(T, plain) === nothing
    end
    # An unregistered aggregate argument is still refused, as RustCall does.
    T = first(types)
    @test_throws RustCall.RustError RustCall.ffi_check_by_value_signature(T, Tuple{Ptr{Nothing},UnregisteredPair})
    # Other result types keep RustCall's own method.
    @test which(RustCall.ffi_check_by_value_signature, Tuple{Type{Int64},Type{plain}}).module === RustCall
end

@testset "bridge calls work through the compile-time check" begin
    ctx = EagerContext()
    x = EagerTensor(ctx, [1.0 2.0; 3.0 4.0])
    @test Array(x .+ x) == [2.0 4.0; 6.0 8.0]
    @test_throws Exception Tenferro.RustCall.result_to_exception(
        Tenferro._rust_call(ctx.graph, Tenferro.RustCrate.copy_array, UInt(10^6), zeros(UInt8, 8), Int32(1)))
    close(ctx)
end
