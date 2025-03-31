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
