using MLX
using Test

@testset "ops" begin
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
    for fn in [:abs], #keys(MLX.Private.unary_ops),
        T in element_types,
        array_size in array_sizes
        N = length(array_size)
        @testset "$fn.(::$MLXArray{$T, $N}), $array_size" begin
            array = ones(T, array_size)
            if N > 2 || N == 0
                mlx_array = MLXArray(array)
            elseif N > 1
                mlx_array = MLXMatrix(array)
            else
                mlx_array = MLXVector(array)
            end
            expected = @eval $fn.($array)
            actual = @eval $fn.($mlx_array)
            if T <: Integer
                @test actual == expected
            else
                @test actual ≈ expected
            end
        end
    end
end
