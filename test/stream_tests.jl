using MLX
using Test

@testset "Stream" begin
    default_device = MLX.default_device()
    @testset "construction" begin
        stream = MLX.Stream(default_device)
    end
    @testset "properties" begin
        stream = MLX.Stream(default_device)

        @test propertynames(stream) == (:device, fieldnames(MLX.Stream)...)

        @test stream.device isa MLX.Device # TODO should test equality, but equality is not implemented for MLX.Device

        @test_throws ArgumentError stream.device = MLX.Device()

        null_stream = MLX.Wrapper.mlx_stream_new()
        @test_throws ArgumentError stream.mlx_stream = null_stream
        MLX.Wrapper.mlx_stream_free(null_stream)
    end
    @testset "equality" begin
        default_stream1 = MLX.default_stream(default_device)
        default_stream2 = MLX.default_stream(default_device)
        @test default_stream1 == default_stream2
    end
    @testset "show" begin
        stream = MLX.Stream(default_device)
        stream_str = repr(stream)
        @test startswith(stream_str, "Stream($(repr(stream.device)), ") &&
            endswith(stream_str, ")")
    end
    @testset "synchronize!" begin
        stream = MLX.Stream(default_device)
        @test MLX.synchronize!(stream) === nothing
    end
    @testset "default_stream" begin
        @test MLX.default_stream(default_device) isa MLX.Stream
    end
    @testset "set_default_stream" begin
        stream = MLX.Stream(default_device)
        @test MLX.set_default_stream(stream) === nothing
        @test MLX.default_stream(default_device) == stream
    end
    @testset "default_cpu_stream" begin
        stream = MLX.default_cpu_stream()
        @test stream.device.type == MLX.DeviceTypeCPU
    end
    if MLX.metal_is_available()
        @testset "default_gpu_stream" begin
            stream = MLX.default_gpu_stream()
            @test stream.device.type == MLX.DeviceTypeGPU
        end
    end
end
