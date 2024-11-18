using MLX
using Test

@testset "metal" begin
    @testset "metal_device_info" begin
        device_info = MLX.metal_device_info()
        @show device_info
    end
end
