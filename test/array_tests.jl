using MLX
using Random
using Test

@testset "MLXArray" begin
    Random.seed!(42)

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
    array_sizes = [(), (1,), (2,), (1, 1), (2, 1), (2, 2), (1, 1, 1)]
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
                @test strides(mlx_array) == strides(array)
                @test Base.elsize(mlx_array) == Base.elsize(array)

                if N > 0
                    @test getindex(mlx_array, 1) == T(1)
                    array[1] = T(1)
                    @test setindex!(mlx_array, T(1), 1) == array
                end

                @test unsafe_wrap(mlx_array) == array

                @testset "similar(::$MLXArray{$T, $N}), array_size=$array_size" begin
                    similar_mlx_array = similar(mlx_array)
                    @test typeof(similar_mlx_array) == typeof(mlx_array)
                    @test size(similar_mlx_array) == size(mlx_array)
                    @test similar_mlx_array !== mlx_array
                end
            end
        end
    end
    @testset "Unsupported Number types" begin
        @test_throws ArgumentError convert(MLX.Wrapper.mlx_dtype, Rational{Int})
    end

    @testset "Broadcasting interface" begin
        for T in element_types, array_size in array_sizes
            N = length(array_size)
            @testset "broadcast(identity, ::$MLXArray{$T, $N}), array_size=$array_size" begin
                array = rand(T, array_size)
                mlx_array = MLXArray(array)

                result = identity.(mlx_array)
                @test result isa MLXArray
                @test result == mlx_array
                @test result !== mlx_array
            end
        end
    end
end
