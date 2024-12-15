mutable struct MLXString <: AbstractString
    mlx_string::Wrapper.mlx_string

    function MLXString(ptr::Ptr{UInt8})
        mlx_string = Wrapper.mlx_string_new_data(ptr)
        this = new(mlx_string)
        finalizer(d -> Wrapper.mlx_string_free(d.mlx_string), this)
        return this
    end
end

MLXString(str::AbstractString) = MLXString(pointer(str))

function Base.ncodeunits(str::MLXString)
    return length(unsafe_string(Wrapper.mlx_string_data(str.mlx_string)))
end

function Base.iterate(str::MLXString)
    return Base.iterate(unsafe_string(Wrapper.mlx_string_data(str.mlx_string)))
end
function Base.iterate(str::MLXString, i::Int)
    return Base.iterate(unsafe_string(Wrapper.mlx_string_data(str.mlx_string)), i)
end

function convert(::Type{String}, str::MLXString)
    return unsafe_string(Wrapper.mlx_string_data(str.mlx_string))
end
