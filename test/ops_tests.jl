using MLX
using Random
using Test

@testset "ops" begin
    Random.seed!(42)

    element_types = MLX.supported_number_types(MLX.DeviceTypeGPU) # TODO Excluding Float64

    array_sizes = [
        # (), # TODO: Excluded broadcasting over 0-dimensional Array: Yields scalar result
        (1,),
        (2,),
        (1, 1),
        (2, 1),
        (2, 2),
        (1, 1, 1),
    ]

    @testset "copy" begin
        for T in element_types, array_size in array_sizes
            N = length(array_size)
            @testset "copy(::$MLXArray{$T, $N}), $array_size" begin
                array = rand(T, array_size)
                if N > 2 || N == 0
                    mlx_array = MLXArray(array)
                elseif N > 1
                    mlx_array = MLXMatrix(array)
                else
                    mlx_array = MLXVector(array)
                end
                actual = copy(mlx_array)
                expected = copy(array)
                @test actual == expected
            end
        end
    end

    @testset "dropdims" begin
        for T in element_types, array_size in array_sizes
            N = length(array_size)
            dims = Dims(unique(rand(1:N, rand(1:N))))
            if !all([array_size[d] == 1 for d in dims])
                continue
            end
            @testset "dropdims(::$MLXArray{$T, $N}; dims = $dims), $array_size" begin
                array = rand(T, array_size)
                if N > 2 || N == 0
                    mlx_array = MLXArray(array)
                elseif N > 1
                    mlx_array = MLXMatrix(array)
                else
                    mlx_array = MLXVector(array)
                end
                actual = dropdims(mlx_array; dims)
                expected = dropdims(array; dims)
                @test actual == expected
            end
        end
    end

    for fn in [:sort] # TODO sortperm is broken
        @testset "$fn" begin
            for T in filter(T -> T != ComplexF32, element_types), # isless is not defined for ComplexF32
                array_size in array_sizes

                N = length(array_size)
                dims = rand(1:N)
                @testset "$fn(::$MLXArray{$T, $N}), $array_size" begin
                    array = rand(T, array_size)
                    if N > 2 || N == 0
                        mlx_array = MLXArray(array)
                    elseif N > 1
                        mlx_array = MLXMatrix(array)
                    else
                        mlx_array = MLXVector(array)
                    end
                    if N == 1
                        actual = @eval $fn($mlx_array)
                        expected = @eval $fn($array)
                    else
                        actual = @eval $fn($mlx_array; dims = $dims)
                        expected = @eval $fn($array; dims = $dims)
                    end
                    @test actual == expected
                end
            end
        end
    end

    @testset "permutedims" begin
        for T in element_types, array_size in array_sizes
            N = length(array_size)
            @testset "permutedims(::$MLXArray{$T, $N}), $array_size" begin
                array = rand(T, array_size)
                if N > 2 || N == 0
                    mlx_array = MLXArray(array)
                elseif N > 1
                    mlx_array = MLXMatrix(array)
                else
                    mlx_array = MLXVector(array)
                end
                actual = permutedims(mlx_array)
                expected = permutedims(array, reverse(1:ndims(array)))
                @test actual == expected
            end
        end
    end

    for (fn, fn_def) in MLX.Private.get_unary_scalar_ops()
        @testset "$fn" begin
            for T in element_types, array_size in array_sizes
                N = length(array_size)
                if !(T <: fn_def.TIn)
                    continue
                end
                @testset "$fn.(::$MLXArray{$T, $N}), $array_size" begin
                    array = rand(T, array_size)
                    array = fn_def.normalize(array, T)
                    if N > 2 || N == 0
                        mlx_array = MLXArray(array)
                    elseif N > 1
                        mlx_array = MLXMatrix(array)
                    else
                        mlx_array = MLXVector(array)
                    end
                    TOut = fn_def.output_type(T)
                    if TOut == Float32 # TODO fn.(array) may return a Float64 array for a Float32 array
                        expected = @eval $TOut.($fn.($array))
                    else
                        expected = @eval $fn.($array)
                    end
                    actual = @eval $fn.($mlx_array)
                    if TOut <: Integer
                        @test actual == expected
                    else
                        @test actual ≈ expected
                    end
                end
            end
        end
    end
end
