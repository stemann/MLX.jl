using MLX
using Test

@testset "array" begin
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
        Float16,
        Float32,
        ComplexF32,
    )
    for T in element_types, N in (0, 1, 2, 3)
        @testset "Array{$T, $N}" begin
            array = repeat([T(1)], N)
            mlx_array = MLXArray(array)
            @test eltype(mlx_array) == T
            @test length(mlx_array) == length(array)
            @test ndims(mlx_array) == ndims(array)
            @test size(mlx_array) == size(array)
            @test strides(mlx_array) == strides(array)
            @test Base.elsize(mlx_array) == Base.elsize(array)
        end
    end
end
