using MLX
using Test

@testset "MLXArray" begin
    @test IndexStyle(MLXArray) == IndexLinear()
    element_types = (
        Bool,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
        # TODO Float16, 
        Float32,
        # TODO Core.BFloat16
        ComplexF32,
    )
    array_sizes = [(), (1,), (2,), (1, 1), (2, 1), (2, 2), (1, 1, 1)]
    for T in element_types, array_size in array_sizes
        N = length(array_size)
        @testset "$MLXArray{$T, $N} $array_size" begin
            array = ones(T, array_size)
            if N > 2 || N == 0
                mlx_array = MLXArray(array)
            elseif N > 1
                mlx_array = MLXMatrix(array)
            else
                mlx_array = MLXVector(array)
            end
            @test eltype(mlx_array) == T
            @test length(mlx_array) == length(array)
            @test ndims(mlx_array) == ndims(array)
            @test size(mlx_array) == size(array)
            @test strides(mlx_array) == strides(array)
            @test Base.elsize(mlx_array) == Base.elsize(array)

            if N > 0
                @test getindex(mlx_array, 1) == T(1)
                array[1] = T(1)
                @test setindex!(mlx_array, T(1), 1) == array
            end

            @test unsafe_wrap(mlx_array) == array
        end
    end
    for T in [Float64]
        @test_throws ArgumentError convert(MLX.Wrapper.mlx_dtype, T)
    end
end
