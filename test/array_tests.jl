using MLX
using Test

@testset "array" begin
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
    for T in element_types, N in (0, 1, 2, 3)
        @testset "Array{$T, $N}" begin
            array = repeat([T(1)], N)
            if array isa AbstractVector
                mlx_array = MLXVector(array)
            elseif array isa AbstractMatrix
                mlx_array = MLXMatrix(array)
            else
                mlx_array = MLXArray(array)
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
