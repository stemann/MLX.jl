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
    if name == :index
        index = Ref(Cint(-1))
        Wrapper.mlx_device_get_index(index, device.mlx_device)
        return Int(index[])
    elseif name == :type
        mlx_device_type = Ref{Wrapper.mlx_device_type}()
        Wrapper.mlx_device_get_type(mlx_device_type, device.mlx_device)
        return DeviceType(UInt32(mlx_device_type[]))
    end
    return getfield(device, name)
end

Base.propertynames(::Device) = (:index, :type, fieldnames(Device)...)

function Base.setproperty!(::Device, name::Symbol, value)
    if name == :index || name == :type
        throw(ArgumentError("Cannot set device $name"))
    end
    throw(ArgumentError("Field $name should not be set"))
end

Base.:(==)(a::Device, b::Device) = Wrapper.mlx_device_equal(a.mlx_device, b.mlx_device) == 1 # TODO mlx_device_equal should return Bool and not Cint

function Base.show(io::IO, device::Device)
    print(io, "MLX.Device(; device_type = MLX.$(device.type), index = $(device.index))")
    return nothing
end

function default_device()
    mlx_device = Ref(Wrapper.mlx_device_new())
    Wrapper.mlx_get_default_device(mlx_device)
    return Device(mlx_device[])
end

function set_default_device(device::Device)
    Wrapper.mlx_set_default_device(device.mlx_device)
    return nothing
end
