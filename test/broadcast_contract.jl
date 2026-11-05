using Test, Tenferro, DifferentiationInterface

@testset "binary broadcast contracts" begin
    for T in (Float32,Float64)
        a = T[0.4 0.7 1.1; 1.3 1.7 2.2]
        b = T[2.3 2.5 2.7]
        da = fill(T(0.2), size(a)); db = fill(T(-0.1), size(b))
        for op in (+,-,*,/,^,rem,max,min)
            f(z) = op.(z[1],z[2])
            for (C,TensorType) in ((EagerContext,EagerTensor),(TracedContext,TracedTensor))
                ctx = C()
                @test Array(f((TensorType(ctx,a),TensorType(ctx,b)))) ≈ f((a,b))
            end
            h = T === Float32 ? T(0.002) : T(1e-5)
            numerical = (f((a+h*da,b+h*db)) - f((a-h*da,b-h*db))) / (2h)
            for backend in (AutoTenferro(),AutoTenferroTraced())
                (dy,) = pushforward(f,backend,(a,b),((da,db),))
                @test dy ≈ numerical rtol=(T === Float32 ? 0.01 : 1e-7) atol=(T === Float32 ? 0.001 : 1e-8)
                ((ga,gb),) = pullback(f,backend,(a,b),(ones(T,size(a)),))
                @test sum(dy) ≈ sum(ga.*da) + sum(gb.*db)
            end
        end
    end
end

@testset "comparison ties" begin
    for backend in (AutoTenferro(),AutoTenferroTraced())
        for op in (maximum,minimum)
            @test gradient(op,backend,[1.0,1,1]) ≈ fill(1/3,3)
        end
        for op in (max,min)
            ga,gb = gradient(z -> sum(op.(z[1],z[2])),backend,([1.0],[1.0]))
            @test ga == [0.5]
            @test gb == [0.5]
        end
    end
end

@testset "integer and Boolean forward scope" begin
    for (C,TensorType) in ((EagerContext,EagerTensor),(TracedContext,TracedTensor))
        for T in (Int32,Int64)
            ctx = C(); a = T[3,-3]; b = T[2,2]
            x,y = TensorType(ctx,a),TensorType(ctx,b)
            for op in (+,-,*,rem,max,min)
                @test Array(op.(x,y)) == op.(a,b)
            end
            @test Array(x ./ y) == a ./ b
            @test eltype(Array(x ./ y)) === eltype(a ./ b)
            @test Array(x .^ y) == a .^ b
        end
        ctx=C(); x=TensorType(ctx,Bool[true,false])
        @test isequal(Array(x ./ x), Bool[true,false] ./ Bool[true,false])
        for op in (+,-,*,^,rem,max,min)
            @test_throws Exception Array(op.(x,x))
        end
    end
end
