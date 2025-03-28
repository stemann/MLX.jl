mutable struct Stream
    mlx_stream::Wrapper.mlx_stream

    function Stream(mlx_stream::Wrapper.mlx_stream)
        this = new(mlx_stream)
        finalizer(s -> Wrapper.mlx_stream_free(s.mlx_stream), this)
        return this
    end
end

function Stream(dev::Device)
    mlx_stream = Wrapper.mlx_stream_new_device(dev.mlx_device)
    return Stream(mlx_stream)
end

function Base.getproperty(stream::Stream, name::Symbol)
    if name == :device
        mlx_device = Ref(Wrapper.mlx_device_new())
        Wrapper.mlx_stream_get_device(mlx_device, stream.mlx_stream)
        return Device(mlx_device[])
    elseif name == :index
        index = Ref(Cint(-1))
        Wrapper.mlx_stream_get_index(index, stream.mlx_stream)
        return Int(index[])
    end
    return getfield(stream, name)
end

Base.propertynames(::Stream) = (:device, :index, fieldnames(Stream)...)

function Base.setproperty!(::Stream, name::Symbol, _)
    if name == :device || name == :index
        throw(ArgumentError("Cannot set stream $name"))
    end
    throw(ArgumentError("Field $name should not be set"))
end

Base.:(==)(a::Stream, b::Stream) = Wrapper.mlx_stream_equal(a.mlx_stream, b.mlx_stream)

function Base.show(io::IO, stream::Stream)
    print(io, "MLX.Stream($(stream.device); index = $(stream.index))")
    return nothing
end

function synchronize!(stream::Stream)
    Wrapper.mlx_synchronize(stream.mlx_stream)
    return nothing
end

function default_stream(dev::Device)
    mlx_stream = Ref(Wrapper.mlx_stream_new())
    Wrapper.mlx_get_default_stream(mlx_stream, dev.mlx_device)
    return Stream(mlx_stream[])
end

function set_default_stream(stream::Stream)
    Wrapper.mlx_set_default_stream(stream.mlx_stream)
    return nothing
end

function default_cpu_stream()
    mlx_stream = Wrapper.mlx_default_cpu_stream_new()
    return Stream(mlx_stream)
end

function default_gpu_stream()
    mlx_stream = Wrapper.mlx_default_gpu_stream_new()
    return Stream(mlx_stream)
end

const stream = ScopedValue{Union{Stream, Nothing}}(nothing)

function get_stream()::Stream
    scoped_stream = stream[]
    if isnothing(scoped_stream)
        return default_stream(get_device())
    end
    return scoped_stream
end
