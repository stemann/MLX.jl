module OpsTests

using MLX
using Random
using Test

include(joinpath(@__DIR__, "TestUtils.jl"))
using .TestUtils

@testset "ops" begin
    Random.seed!(42)

    array_sizes = [(), (1,), (2,), (1, 1), (2, 1), (2, 2), (1, 1, 1)]

    testset_foreach("copy"; array_sizes) do T, array_size
        array = rand(T, array_size)
        @test copy(to_mlx(array)) == copy(array)
    end

    testset_foreach("dropdims"; array_sizes) do T, array_size
        N = length(array_size)
        N == 0 && return nothing # dropdims fails for 0-dimensional Array
        dims = Dims(unique(rand(1:N, rand(1:N))))
        all(array_size[d] == 1 for d in dims) || return nothing
        array = rand(T, array_size)
        @test dropdims(to_mlx(array); dims) == dropdims(array; dims)
    end

    # isless is not defined for complex numbers
    real_types(device_type) =
        filter(T -> T <: Real, MLX.supported_number_types(device_type))
    for fn in (sort, sortperm)
        testset_foreach("$fn"; element_types = real_types, array_sizes) do T, array_size
            N = length(array_size)
            N == 0 && return nothing # A 0-dimensional array has no dims to sort along
            array = rand(T, array_size)
            if N == 1
                @test fn(to_mlx(array)) == fn(array)
            else
                dims = rand(1:N)
                @test fn(to_mlx(array); dims) == fn(array; dims)
            end
        end
    end

    testset_foreach("permutedims"; array_sizes) do T, array_size
        N = length(array_size)
        array = rand(T, array_size)
        mlx_array = to_mlx(array)
        perm = Tuple(randperm(N))
        actual = permutedims(mlx_array, perm)
        @test actual isa MLXArray{T, N}
        @test actual == permutedims(array, perm)
        if N == 2
            @test permutedims(mlx_array) isa MLXMatrix{T}
            @test permutedims(mlx_array) == permutedims(array)
        end
    end

    # Integers from a small range, so they are exactly representable as Float32 inputs
    test_values(T) =
        T <: Integer && T != Bool ? (T <: Signed ? (T(-10):T(10)) : (T(0):T(10))) : T

    # MLX computes some Float64 functions with about Float32 precision
    rtol = sqrt(eps(Float32))

    for (fn_symbol, fn_def) in MLX.Private.get_unary_scalar_ops()
        fn = eval(fn_symbol)
        function input_types(device_type)
            return filter(T -> T <: fn_def.TIn, MLX.supported_number_types(device_type))
        end

        testset_foreach(
            "$fn.(::MLXArray)"; element_types = input_types, array_sizes
        ) do T, array_size
            N = length(array_size)
            array = fn_def.normalize(rand(test_values(T), array_size), T)
            if N == 0 # Broadcasting over a 0-dimensional Array yields a scalar
                array = fill(only(array))
            end
            TOut = fn_def.output_type(T)
            # Julia returns Float64 for integers, MLX Float32
            expected = TOut == Float32 ? TOut.(fn.(array)) : fn.(array)
            actual = fn.(to_mlx(array))
            @test actual isa (N == 0 ? MLXNumber{TOut} : MLXArray{TOut, N})
            @test convert(Number, MLX.Wrapper.mlx_array_dtype(actual)) == TOut
            if TOut <: Integer
                @test actual == expected
            elseif N == 0
                @test isapprox(TOut(actual), expected; rtol)
            else
                @test isapprox(actual, expected; rtol)
            end
        end

        testset_foreach_type("$fn(::MLXNumber)"; element_types = input_types) do T
            value = only(fn_def.normalize([rand(test_values(T))], T))
            TOut = fn_def.output_type(T)
            expected = fn(value)
            actual = fn(MLXNumber(value))
            if fn in (isfinite, isinf, isnan)
                @test actual === expected
            else
                @test actual isa MLXNumber{TOut}
                @test convert(Number, MLX.Wrapper.mlx_array_dtype(actual)) == TOut
                if TOut <: Integer
                    @test actual == expected
                else
                    @test isapprox(TOut(actual), expected; rtol)
                end
            end
        end
    end
end

end
