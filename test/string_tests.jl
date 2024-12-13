using MLX
using Test

@testset "string" begin
    @testset "constructor" begin
        s = ""
        s_mlx = MLXString(s)
        @test s_mlx == s
    end
end
