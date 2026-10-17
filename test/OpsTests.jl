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
end

end
