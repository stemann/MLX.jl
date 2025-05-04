@enum DeviceType::UInt32 begin
    DeviceTypeCPU = 0
    DeviceTypeGPU = 1
end

function supported_number_types(device_type::DeviceType = DeviceTypeCPU)
    types = [
        Bool,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
        # TODO Float16, 
        Float32,
        # TODO Core.BFloat16
        ComplexF32,
    ]
    if device_type == DeviceTypeCPU
        return vcat(types, [Float64])
    elseif device_type == DeviceTypeGPU
        return types
    else
        throw(ArgumentError("Unsupported device type: $device_type"))
    end
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

const device = ScopedValue{Union{Device, Nothing}}(nothing)

function get_device()::Device
    scoped_device = device[]
    if isnothing(scoped_device)
        scoped_stream = stream[]
        if isnothing(scoped_stream)
            return default_device()
        end
        return scoped_stream.device
    end
    return scoped_device
end
