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
end

end
