module ArrayTests

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

    array_sizes = [(), (1,), (2,), (1, 1), (2, 1), (3, 2), (4, 3, 2)]

    # Calls f(T, array_size) in a test set named name, with nested test sets per device,
    # element type and array size
    function testset_foreach(f, name; element_types = MLX.supported_number_types)
        @testset "$name" begin
            @testset "$name, $device_type" for device_type in device_types
                with(MLX.device => MLX.Device(; device_type)) do
                    prefix = "$name, $device_type"
                    @testset "$prefix, $MLXArray{$T}" for T in element_types(device_type)
                        for array_size in array_sizes
                            N = length(array_size)
                            @testset "$prefix, $MLXArray{$T, $N}, array_size=$array_size" begin
                                f(T, array_size)
                            end
                        end
                    end
                end
            end
        end
    end

    function to_mlx(array::AbstractArray)
        if ndims(array) == 1
            return MLXVector(array)
        elseif ndims(array) == 2
            return MLXMatrix(array)
        end
        return MLXArray(array)
    end

    @testset "AbstractArray interface" begin
        testset_foreach("Type parameters") do T, array_size
            N = length(array_size)
            mlx_array = to_mlx(ones(T, array_size))
            @test mlx_array isa AbstractArray{T, N}
            @test eltype(mlx_array) == T
            @test ndims(mlx_array) == N
        end
        testset_foreach("Required methods") do T, array_size
            array = rand(T, array_size)
            mlx_array = to_mlx(array)
            @test size(mlx_array) == size(array)
            if ndims(array) > 0
                @test getindex(mlx_array, 1) == array[1]
            end
        end
        @testset "Optional methods" begin
            @testset "IndexStyle" begin
                @test IndexStyle(MLXArray) == IndexLinear()
            end
            testset_foreach("length, iterate and setindex!") do T, array_size
                array = rand(T, array_size)
                mlx_array = to_mlx(array)
                @test length(mlx_array) == length(array)
                @test collect(mlx_array) == array
                if ndims(array) > 0
                    array[1] = iszero(array[1]) ? one(T) : zero(T) # Differs from array[1]
                    @test setindex!(mlx_array, array[1], 1) == array
                end
            end
            testset_foreach("UndefInitializer") do T, array_size
                N = length(array_size)
                for mlx_array in
                    (MLXArray{T, N}(undef, array_size), MLXArray{T}(undef, array_size))
                    @test mlx_array isa MLXArray{T, N}
                    @test size(mlx_array) == array_size
                end
            end
            testset_foreach("similar") do T, array_size
                N = length(array_size)
                S = T == Float32 ? Int32 : Float32 # Ensure S != T
                other_dims = (3, 2)
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
    testset_foreach("Strided array interface") do T, array_size
        array = rand(T, array_size)
        mlx_array = MLXArray(array)

        if ndims(array) > 0
            @test strides(mlx_array) ==
                reverse(strides(permutedims(array, reverse(1:ndims(array)))))
        else
            @test strides(mlx_array) == strides(array)
        end
        @test Base.unsafe_convert(Ptr{T}, mlx_array) isa Ptr{T}
        @test unsafe_wrap(mlx_array) == array
        @test Base.elsize(mlx_array) == Base.elsize(array)
        @test Base.elsize(typeof(mlx_array)) == Base.elsize(typeof(array))

        another_mlx_array = MLXArray(mlx_array)
        @test another_mlx_array == mlx_array
        @test strides(another_mlx_array) == strides(mlx_array)
    end
    @testset "Unsupported Number types" begin
        @test_throws ArgumentError convert(MLX.Wrapper.mlx_dtype, Rational{Int})
    end

    testset_foreach("BitArray"; element_types = _ -> [Bool]) do T, array_size
        array = BitArray(rand(Bool, array_size))
        @test to_mlx(array) == array
    end
end

end
