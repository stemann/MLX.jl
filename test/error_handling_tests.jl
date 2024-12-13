using MLX
using Test

@testset "error_handling" begin
    @test_throws MLXException MLX.Wrapper.mlx_string_data(MLX.Wrapper.mlx_string_new())
end
