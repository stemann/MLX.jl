module OpsTests

using MLX
using Random
using Test

include(joinpath(@__DIR__, "TestUtils.jl"))
using .TestUtils

@testset "ops" begin
    Random.seed!(42)

    testset_foreach("copy") do T, array_size
        array = rand(T, array_size)
        mlx_array = to_mlx(array)
        expected = copy(array)
        actual = copy(mlx_array)
        @test actual == expected
    end

    testset_foreach("dropdims") do T, array_size
        N = length(array_size)
        N == 0 && return nothing # dropdims fails for 0-dimensional Array
        dims = Dims(unique(rand(1:N, rand(1:N))))
        all(array_size[d] == 1 for d in dims) || return nothing
        array = rand(T, array_size)
        mlx_array = to_mlx(array)
        expected = dropdims(array; dims)
        actual = dropdims(mlx_array; dims)
        @test actual == expected
    end

    # isless is not defined for complex numbers, and MLX 0.23.2 does not support sorting Bool on
    # the GPU
    function sortable_types(device_type)
        return filter(MLX.supported_number_types(device_type)) do T
            return T <: Real && !(T == Bool && device_type == MLX.DeviceTypeGPU)
        end
    end
    for fn in (sort, sortperm)
        testset_foreach("$fn"; element_types = sortable_types) do T, array_size
            N = length(array_size)
            N == 0 && return nothing # A 0-dimensional array has no dims to sort along
            array = rand(T, array_size)
            mlx_array = to_mlx(array)
            if N == 1
                expected = fn(array)
                actual = fn(mlx_array)
            else
                dims = rand(1:N)
                expected = fn(array; dims)
                actual = fn(mlx_array; dims)
            end
            @test actual == expected
        end
    end

    testset_foreach("permutedims") do T, array_size
        N = length(array_size)
        array = rand(T, array_size)
        mlx_array = to_mlx(array)
        perm = Tuple(randperm(N))
        @static if VERSION < v"1.11" # permutedims(::Array{T, 0}, ()) fails on Julia 1.10
            expected = N == 0 ? copy(array) : permutedims(array, perm)
        else
            expected = permutedims(array, perm)
        end
        actual = permutedims(mlx_array, perm)
        @test actual isa MLXArray{T, N}
        @test actual == expected
        if N == 2
            expected = permutedims(array)
            actual = permutedims(mlx_array)
            @test actual isa MLXMatrix{T}
            @test actual == expected
        end
    end

    # Integers from a small range, so they are exactly representable as Float32 inputs
    test_values(T) =
        T <: Integer && T != Bool ? (T <: Signed ? (T(-10):T(10)) : (T(0):T(10))) : T

    # MLX 0.23.2 computes some Float64 functions with about Float32 precision
    rtol = sqrt(eps(Float32))

    for (fn_symbol, fn_def) in MLX.Internals.get_unary_ops()
        fn = eval(fn_symbol)
        function input_types(device_type)
            return filter(
                T -> T <: fn_def.test_type(device_type),
                MLX.supported_number_types(device_type),
            )
        end
        testset_foreach("$fn"; element_types = input_types) do T, array_size
            N = length(array_size)
            array = fn_def.test_input(rand(test_values(T), array_size), T)
            if N == 0 # Broadcasting over a 0-dimensional Array yields a scalar
                array = fill(only(array))
            end
            mlx_array = to_mlx(array)
            TOut = fn_def.output_type(T)
            if T <: Integer # Base returns Float64 for integer inputs, where MLX 0.23.2 returns Float32
                expected = TOut.(fn.(array))
            else
                expected = fn.(array)
            end
            actual = fn.(mlx_array)
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
            value = only(fn_def.test_input([rand(test_values(T))], T))
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

    # Mixed element types where MLX's own type promotion differs from Julia's, and some where it
    # agrees
    mixed_types = Dict(
        Bool => [Int8],
        Int8 => [UInt8],
        Int32 => [Int64],
        UInt64 => [Int64],
        Int64 => [Float32],
        Float32 => [ComplexF32, Float64],
    )
    # Sizes of the operands, nothing for a scalar
    size_pairs = [
        ((), ()),
        ((2,), nothing),
        (nothing, (2,)),
        ((2,), ()),
        ((2,), (2,)),
        ((3, 2), (3,)),
        ((3,), (3, 2)),
        ((3, 2), (1, 2)),
        ((3, 2), (3, 1)),
        ((4, 3, 2), (4,)),
        ((4, 3, 2), nothing),
    ]
    test_operand(T, size) =
        isnothing(size) ? rand(test_values(T)) : rand(test_values(T), size)
    to_mlx_operand(x) = x isa Number ? x : to_mlx(x)

    for (fn_symbol, fn_def) in MLX.Internals.get_binary_ops()
        fn = eval(fn_symbol)
        function first_types(device_type)
            return filter(
                T -> T <: fn_def.test_type(device_type),
                MLX.supported_number_types(device_type),
            )
        end
        testset_foreach_type(
            "$fn.(::MLXArray, ::MLXArray)"; element_types = first_types
        ) do T
            device_type = MLX.get_device().type
            second_types = filter(
                S ->
                    S <: fn_def.test_type(device_type) &&
                        S in MLX.supported_number_types(device_type),
                [T; get(mixed_types, T, [])],
            )
            for S in second_types, (a_size, b_size) in size_pairs
                @testset "$fn.(::$T, ::$S), sizes $a_size, $b_size" begin
                    a, b = fn_def.test_input(
                        test_operand(T, a_size), test_operand(S, b_size)
                    )
                    C = if isnothing(fn_def.compute_type)
                        Base.promote_op(fn, T, S)
                    else
                        fn_def.compute_type(T, S)
                    end
                    N = max(ndims(a), ndims(b))
                    expected = C.(fn.(a, b)) # e.g. Float32 rather than Float64 for integer division
                    actual = fn.(to_mlx_operand(a), to_mlx_operand(b))
                    @test actual isa (N == 0 ? MLXNumber{C} : MLXArray{C, N})
                    @test convert(Number, MLX.Wrapper.mlx_array_dtype(actual)) == C
                    if C <: Integer
                        @test actual == expected
                    elseif N == 0
                        @test isapprox(C(actual), expected; rtol)
                    else
                        @test isapprox(actual, expected; rtol)
                    end
                end
            end
            # Julia's broadcasting does not align (3, 2) and (2,)
            @test_throws DimensionMismatch fn.(to_mlx(rand(T, 3, 2)), to_mlx(rand(T, 2)))
        end
    end

    @testset "Results that differ from ops on Array" begin
        # Float32 where ops on Array return Float64, which MLX does not support on GPUs
        @test sin.(MLXArray([1, 2])) isa MLXVector{Float32}
        @test MLXArray([1, 2]) / 2 isa MLXVector{Float32}
        @test MLXArray([1, 2]) ./ MLXArray([2, 4]) isa MLXVector{Float32}
        # Ops on Array return ComplexF64, which MLX 0.23.2 does not support
        @test_throws ArgumentError MLXArray(ComplexF32[1, 2]) .* 2.0
    end

    # Base's conj, real, imag and - on arrays broadcast, and keep 0-dim arrays
    unary_ops = MLX.Internals.get_unary_ops()
    testset_foreach("0-dim array functions"; array_sizes = [()]) do T, array_size
        for fn in (conj, real, imag, -)
            T <: unary_ops[Symbol(fn)].input_type || continue
            @testset "$fn(::$MLXArray{$T, 0})" begin
                array = rand(T, array_size)
                mlx_array = to_mlx(array)
                expected = fn(array)
                actual = fn(mlx_array)
                @test actual isa MLXArray{eltype(expected), 0}
                @test actual == expected
            end
        end
    end
end

end
