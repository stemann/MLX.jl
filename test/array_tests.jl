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
        Float64,
        # TODO Core.BFloat16
        ComplexF32,
    )
    array_sizes = [(), (1,), (2,), (1, 1), (2, 1), (3, 2), (4, 3, 2)]
    @testset "AbstractArray interface" begin
        for T in element_types, array_size in array_sizes
            N = length(array_size)
            @testset "$MLXArray{$T, $N}, array_size=$array_size" begin
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

                if N > 0
                    @test getindex(mlx_array, 1) == T(1)
                    array[1] = T(1)
                    @test setindex!(mlx_array, T(1), 1) == array
                end
            end
        end
    end
    @testset "Strided array interface" begin
        for T in element_types, array_size in array_sizes
            N = length(array_size)
            @testset "$MLXArray{$T, $N}, array_size=$array_size" begin
                array = ones(T, array_size)
                mlx_array = MLXArray(array)

                @test strides(mlx_array) == reverse(strides(permutedims(array, reverse(1:ndims(array)))))
                @test Base.unsafe_convert(Ptr{T}, mlx_array) isa Ptr{T}
                @test unsafe_wrap(mlx_array) == array
                @test Base.elsize(mlx_array) == Base.elsize(array)

                another_mlx_array = MLXArray(mlx_array)
                @test another_mlx_array == mlx_array
                @test strides(another_mlx_array) == strides(mlx_array)
            end
        end
        for T in element_types
            @test Base.elsize(MLXArray{T, 0}) == Base.elsize(Array{T, 0})
        end
    end
    @testset "Unsupported Number types" begin
        @test_throws ArgumentError convert(MLX.Wrapper.mlx_dtype, Rational{Int})
    end
end
