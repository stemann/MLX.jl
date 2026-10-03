@static if VERSION < v"1.11"
    using ScopedValues
else
    using Base.ScopedValues
end

using MLX
using Test

@testset "MLXArray" begin
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
            end
        end
    end
    @testset "UndefInitializer" begin
        for device_type in device_types,
            T in MLX.supported_number_types(device_type),
            array_size in array_sizes

            N = length(array_size)
            @testset "$MLXArray{$T, $N}, array_size=$array_size, $device_type" begin
                with(MLX.device => MLX.Device(; device_type)) do
                    @testset "MLXArray{T, N}(undef, dims)" begin
                        mlx_array = MLXArray{T, N}(undef, array_size)
                        @test mlx_array isa MLXArray{T, N}
                        @test size(mlx_array) == array_size
                    end
                    @testset "MLXArray{T}(undef, dims)" begin
                        mlx_array = MLXArray{T}(undef, array_size)
                        @test mlx_array isa MLXArray{T, N}
                        @test size(mlx_array) == array_size
                    end
                end
            end
        end
    end
    @testset "similar" begin
        other_dims = (3, 2)

        for device_type in device_types,
            T in MLX.supported_number_types(device_type),
            array_size in array_sizes

            N = length(array_size)
            S = T == Float32 ? Int32 : Float32 # Ensure S != T
            @testset "$MLXArray{$T, $N}, array_size=$array_size, $device_type" begin
                with(MLX.device => MLX.Device(; device_type)) do
                    @testset "similar(::Type{MLXArray{T}}, dims)" begin
                        result = similar(MLXArray{T}, array_size)
                        @test result isa MLXArray{T, N}
                        @test size(result) == array_size
                    end

                    mlx_array = MLXArray(ones(T, array_size))
                    @testset "similar(a)" begin
                        result = similar(mlx_array)
                        @test result isa MLXArray{T, N}
                        @test size(result) == array_size
                        @test pointer(result) != pointer(mlx_array)
                    end
                    @testset "similar(a, S)" begin
                        result = similar(mlx_array, S)
                        @test result isa MLXArray{S, N}
                        @test size(result) == array_size
                        @test pointer(result) != pointer(mlx_array)
                    end
                    @testset "similar(a, dims)" begin
                        result = similar(mlx_array, other_dims)
                        @test result isa MLXArray{T, length(other_dims)}
                        @test size(result) == other_dims
                        @test pointer(result) != pointer(mlx_array)
                    end
                    @testset "similar(a, S, dims)" begin
                        result = similar(mlx_array, S, other_dims)
                        @test result isa MLXArray{S, length(other_dims)}
                        @test size(result) == other_dims
                        @test pointer(result) != pointer(mlx_array)
                    end
                end
            end
        end
    end
    @testset "Strided array interface" begin
        for device_type in device_types,
            T in MLX.supported_number_types(device_type),
            array_size in array_sizes

            N = length(array_size)
            @testset "$MLXArray{$T, $N}, array_size=$array_size, $device_type" begin
                with(MLX.device => MLX.Device(; device_type)) do
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
        end
        for T in MLX.supported_number_types()
            @test Base.elsize(MLXArray{T, 0}) == Base.elsize(Array{T, 0})
        end
    end
    @testset "Unsupported Number types" begin
        @test_throws ArgumentError convert(MLX.Wrapper.mlx_dtype, Rational{Int})
    end
end
