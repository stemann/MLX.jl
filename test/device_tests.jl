using MLX
using Test

@testset "Device" begin
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
        device = MLX.Device(; device_type = MLX.DeviceTypeGPU, index = 1)

        @test propertynames(device) == (:index, :type, fieldnames(MLX.Device)...)

        @test device.index == 1
        @test device.type == MLX.DeviceTypeGPU

        @test_throws ArgumentError device.index = 2
        @test_throws ArgumentError device.type = MLX.DeviceTypeCPU

        null_device = MLX.Wrapper.mlx_device_new()
        @test_throws ArgumentError device.mlx_device = null_device
        MLX.Wrapper.mlx_device_free(null_device)
    end
    @testset "equality" begin
        default_device1 = MLX.default_device()
        default_device2 = MLX.default_device()
        @test default_device1 == default_device2
    end
    @testset "show" begin
        device = MLX.Device(; device_type = MLX.DeviceTypeCPU)
        @test eval(Meta.parse(repr(device))) == device
        device = MLX.Device(; device_type = MLX.DeviceTypeGPU)
        @test eval(Meta.parse(repr(device))) == device
    end
    @testset "default_device" begin
        default_device = MLX.default_device()
        if MLX.metal_is_available()
            @test default_device.type == MLX.DeviceTypeGPU
        else
            @test default_device.type == MLX.DeviceTypeCPU
        end
    end
    @testset "set_default_device" begin
        device = MLX.Device()
        @test MLX.set_default_device(device) === nothing
        @test repr(MLX.default_device()) == repr(device)
    end
end
