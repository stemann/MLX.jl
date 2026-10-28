module NumberTypesTests

using MLX
using Test

@testset "Number types" begin
    @testset "Conversion to mlx_dtype and back, $T" for T in MLX.supported_number_types()
        @test convert(Number, convert(MLX.Wrapper.mlx_dtype, T)) == T
    end
    @testset "Unsupported Number types" begin
        @test_throws ArgumentError convert(MLX.Wrapper.mlx_dtype, Rational{Int})
        @test_throws ArgumentError convert(Number, MLX.Wrapper.mlx_dtype(typemax(UInt32)))
    end
end

end
