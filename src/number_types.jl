function Base.convert(::Type{Wrapper.mlx_dtype}, type::Type{<:Number})
    if type == Bool
        return Wrapper.MLX_BOOL
    elseif type == UInt8
        return Wrapper.MLX_UINT8
    elseif type == UInt16
        return Wrapper.MLX_UINT16
    elseif type == UInt32
        return Wrapper.MLX_UINT32
    elseif type == UInt64
        return Wrapper.MLX_UINT64
    elseif type == Int8
        return Wrapper.MLX_INT8
    elseif type == Int16
        return Wrapper.MLX_INT16
    elseif type == Int32
        return Wrapper.MLX_INT32
    elseif type == Int64
        return Wrapper.MLX_INT64
    elseif type == Float16
        return Wrapper.MLX_FLOAT16
    elseif type == Float32
        return Wrapper.MLX_FLOAT32
    elseif type == Float64
        return Wrapper.MLX_FLOAT64
        # TODO Handle Wrapper.MLX_BFLOAT16
    elseif type == ComplexF32
        return Wrapper.MLX_COMPLEX64
    else
        throw(ArgumentError("Unsupported type: $type"))
    end
end

function Base.convert(::Type{<:Number}, dtype::Wrapper.mlx_dtype)
    if dtype == Wrapper.MLX_BOOL
        return Bool
    elseif dtype == Wrapper.MLX_UINT8
        return UInt8
    elseif dtype == Wrapper.MLX_UINT16
        return UInt16
    elseif dtype == Wrapper.MLX_UINT32
        return UInt32
    elseif dtype == Wrapper.MLX_UINT64
        return UInt64
    elseif dtype == Wrapper.MLX_INT8
        return Int8
    elseif dtype == Wrapper.MLX_INT16
        return Int16
    elseif dtype == Wrapper.MLX_INT32
        return Int32
    elseif dtype == Wrapper.MLX_INT64
        return Int64
    elseif dtype == Wrapper.MLX_FLOAT16
        return Float16
    elseif dtype == Wrapper.MLX_FLOAT32
        return Float32
    elseif dtype == Wrapper.MLX_FLOAT64
        return Float64
        # TODO Handle Wrapper.MLX_BFLOAT16
    elseif dtype == Wrapper.MLX_COMPLEX64
        return ComplexF32
    else
        throw(ArgumentError("Unsupported mlx_dtype: $dtype"))
    end
end
