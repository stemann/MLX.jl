@static if VERSION < v"1.11"
    using ScopedValues
else
    using Base.ScopedValues
end

using MLX
using Random
using Test

@testset "MLXArray" begin
    Random.seed!(42)

    device_types = [MLX.DeviceTypeCPU]
    if MLX.metal_is_available()
        push!(device_types, MLX.DeviceTypeGPU)
    end

    @test IndexStyle(MLXArray) == IndexLinear()

    array_sizes = [(), (1,), (2,), (1, 1), (2, 1), (3, 2), (4, 3, 2)]

    @testset "AbstractArray interface" begin
        element_types = MLX.supported_number_types()

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

                @testset "similar(::$MLXArray{$T, $N}), array_size=$array_size" begin
                    for device_type in device_types
                        if T ∉ MLX.supported_number_types(device_type)
                            continue
                        end
                        @testset "similar(::$MLXArray{$T, $N}), with array_size=$array_size, $device_type" begin
                            with(MLX.device => MLX.Device(; device_type)) do
                                similar_mlx_array = similar(mlx_array)
                                @test typeof(similar_mlx_array) == typeof(mlx_array)
                                @test size(similar_mlx_array) == size(mlx_array)
                                @test similar_mlx_array !== mlx_array
                            end
                        end
                    end
                end
            end
        end
    end
    @testset "Strided array interface" begin
        element_types = MLX.supported_number_types(MLX.DeviceTypeGPU) # TODO Excluding Float64

        for T in element_types, array_size in array_sizes
            N = length(array_size)
            @testset "$MLXArray{$T, $N}, array_size=$array_size" begin
                array = ones(T, array_size)
                mlx_array = MLXArray(array)

                if N > 0
                    @test strides(mlx_array) ==
                        reverse(strides(permutedims(array, reverse(1:ndims(array)))))
                else
                    @test strides(mlx_array) == strides(array)
                end
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

    @testset "Broadcasting interface" begin
        for device_type in device_types,
            T in MLX.supported_number_types(device_type),
            array_size in array_sizes

            N = length(array_size)
            @testset "broadcast(identity, ::$MLXArray{$T, $N}), array_size=$array_size, $device_type" begin
                array = rand(T, array_size)
                mlx_array = MLXArray(array)

                with(MLX.device => MLX.Device(; device_type)) do
                    result = identity.(mlx_array)
                    @test result isa MLXArray
                    @test result == mlx_array
                    @test result !== mlx_array
                end
            end
        end
    end
end
