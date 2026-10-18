module TransformsTests

using MLX
using Test

include(joinpath(@__DIR__, "TestUtils.jl"))
using .TestUtils

@testset "evaluate" begin
    testset_foreach_type("evaluate"; element_types = _ -> [Float32]) do T
        a = MLXArray(T[1, 2, 3])
        b = MLXArray(T[4, 5, 6])
        c = a + b
        d = c .* a
        x = -MLXNumber(T(5))
        @test !MLX.Internals.is_available(c)
        @test !MLX.Internals.is_available(d)
        @test !MLX.Internals.is_available(x)
        MLX.evaluate(d, x)
        @test MLX.Internals.is_available(d)
        @test MLX.Internals.is_available(x)
        @test d == T[5, 14, 27]
        @test x == -5
        MLX.evaluate()
    end
end

end
