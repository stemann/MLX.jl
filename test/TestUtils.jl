module TestUtils

@static if VERSION < v"1.11"
    using ScopedValues
else
    using Base.ScopedValues
end

using MLX
using Test

export array_sizes, device_types, testset_foreach, testset_foreach_type, to_mlx

const device_types =
    MLX.metal_is_available() ? [MLX.DeviceTypeCPU, MLX.DeviceTypeGPU] : [MLX.DeviceTypeCPU]

const array_sizes = [(), (1,), (2,), (1, 1), (2, 1), (3, 2), (4, 3, 2)]

# Calls f(T) in a test set named name, with nested test sets per device and element type
function testset_foreach_type(f, name; element_types = MLX.supported_number_types)
    @testset "$name" begin
        @testset "$name, $device_type" for device_type in device_types
            with(MLX.device => MLX.Device(; device_type)) do
                @testset "$name, $device_type, $T" for T in element_types(device_type)
                    f(T)
                end
            end
        end
    end
end

# Calls f(T, array_size) in a test set named name, with nested test sets per device,
# element type and array size
function testset_foreach(
    f, name; element_types = MLX.supported_number_types, array_sizes = TestUtils.array_sizes
)
    @testset "$name" begin
        @testset "$name, $device_type" for device_type in device_types
            with(MLX.device => MLX.Device(; device_type)) do
                prefix = "$name, $device_type"
                @testset "$prefix, $MLXArray{$T}" for T in element_types(device_type)
                    for array_size in array_sizes
                        N = length(array_size)
                        @testset "$prefix, $MLXArray{$T, $N}, array_size=$array_size" begin
                            f(T, array_size)
                        end
                    end
                end
            end
        end
    end
end

function to_mlx(array::AbstractArray)
    if ndims(array) == 1
        return MLXVector(array)
    elseif ndims(array) == 2
        return MLXMatrix(array)
    end
    return MLXArray(array)
end

end
