@enum DeviceType::UInt32 begin
    DeviceTypeCPU = 0
    DeviceTypeGPU = 1
end

mutable struct Device
    mlx_device::Wrapper.mlx_device

    function Device(mlx_device::Wrapper.mlx_device)
        this = new(mlx_device)
        finalizer(d -> Wrapper.mlx_device_free(d.mlx_device), this)
        return this
    end
end

function Device(; device_type = DeviceTypeCPU, index::Int = 0)
    mlx_device_type = Wrapper.mlx_device_type(UInt32(device_type))
    mlx_device = Wrapper.mlx_device_new_type(mlx_device_type, index)
    return Device(mlx_device)
end

function Base.getproperty(device::Device, name::Symbol)
    if name == :type
        mlx_device_type = Wrapper.mlx_device_get_type(device.mlx_device)
        return DeviceType(UInt32(mlx_device_type))
    end
    return getfield(device, name)
end

Base.propertynames(::Device) = (fieldnames(Device)..., :type)

function Base.setproperty!(device::Device, name::Symbol, value)
    if name == :type
        throw(ArgumentError("Cannot set device type"))
    end
    throw(ArgumentError("Cannot set field $name"))
end

# TODO This should return, e.g., "Device(; device_type = DeviceTypeCPU, index = 0)", and not "Device(cpu, 0)" (the result from mlx_device_tostring)
function Base.show(io::IO, device::Device)
    mlx_str = Wrapper.mlx_string_new_data(pointer("")) # Workaround for mlx_string_new()
    Wrapper.mlx_device_tostring(Ref(mlx_str), device.mlx_device)
    str = unsafe_string(Wrapper.mlx_string_data(mlx_str))
    Wrapper.mlx_string_free(mlx_str)
    print(io, str)
    return nothing
end
