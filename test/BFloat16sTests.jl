module BFloat16sTests

using BFloat16s
using MLX
using Test

include(joinpath(@__DIR__, "TestUtils.jl"))
using .TestUtils

@testset "BFloat16s extension" begin
    @testset "Conversion to mlx_dtype and back" begin
        @test convert(Number, convert(MLX.Wrapper.mlx_dtype, BFloat16)) == BFloat16
    end
    testset_foreach("MLXArray"; element_types = _ -> [BFloat16]) do T, array_size
        N = length(array_size)
        array = rand(T, array_size)
        mlx_array = to_mlx(array)
        @test mlx_array isa MLXArray{T, N}
        @test collect(mlx_array) == array
        @test collect(mlx_array .+ mlx_array) == collect(array .+ array)
        if N > 0
            mlx_array[1] = array[1] = array[1] + one(T)
            @test collect(mlx_array) == array
        end
    end
end

end
