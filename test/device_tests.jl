using MLX
using Test

@testset "device" begin
    @testset "get_default_device" begin
        device = MLX.get_default_device()
        @show device
    end
end
