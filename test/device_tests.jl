using MLX
using Test

@testset "device" begin
    @testset "construction" begin
        for kwargs in [
            NamedTuple(),
            (; device_type = MLX.DeviceTypeCPU),
            (; device_type = MLX.DeviceTypeGPU),
        ]
            @testset "construction with $kwargs" begin
                expected_device_type = get(kwargs, :device_type, MLX.DeviceTypeCPU)
                device = MLX.Device(; kwargs...)
                @test device.type == expected_device_type
            end
        end
    end
    @testset "properties" begin
        device = MLX.Device(; device_type = MLX.DeviceTypeGPU)

        @test propertynames(device) == (:mlx_device, :type)

        @test device.type == MLX.DeviceTypeGPU

        null_device = MLX.Wrapper.mlx_device_new()
        @test_throws ArgumentError device.mlx_device = null_device
        MLX.Wrapper.mlx_device_free(null_device)

        @test_throws ArgumentError device.type = MLX.DeviceTypeCPU
    end
    @testset "show" begin
        device = MLX.Device(; device_type = MLX.DeviceTypeCPU)
        @test repr(device) == "Device(cpu, 0)"
        device = MLX.Device(; device_type = MLX.DeviceTypeGPU)
        @test repr(device) == "Device(gpu, 0)"
    end
end
