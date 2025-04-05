mutable struct MLXArray{T, N} <: AbstractArray{T, N}
    mlx_array::Wrapper.mlx_array

    function MLXArray{T, N}(mlx_array::Wrapper.mlx_array) where {T, N}
        this = new(mlx_array)
        finalizer(d -> Wrapper.mlx_array_free(d.mlx_array), this)
        return this
    end
end

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
        return Wrapper.MLX_COMPLEX64 # MLX_COMPLEX64 is a complex of Float32
    else
        throw(ArgumentError("Unsupported type: $type"))
    end
end

function MLXArray{T, N}(array::AbstractArray{T, N}) where {T, N}
    shape = Ref(Cint.(reverse(size(array))))
    dtype = convert(Wrapper.mlx_dtype, T)
    mlx_array = GC.@preserve array shape Wrapper.mlx_array_new_data(
        pointer(array), shape, N, dtype
    )
    return MLXArray{T, N}(mlx_array)
end

function MLXArray(array::AbstractArray{T, N}) where {T, N}
    return MLXArray{T, N}(array)
end

const MLXVector{T} = MLXArray{T, 1}
MLXVector(array::AbstractVector{T}) where {T} = MLXVector{T}(array)

const MLXMatrix{T} = MLXArray{T, 2}
MLXMatrix(array::AbstractMatrix{T}) where {T} = MLXMatrix{T}(array)

const MLXVecOrMat{T} = Union{MLXVector{T}, MLXMatrix{T}}

# AbstractArray interface, cf. https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

function Base.size(array::MLXArray)
    return Tuple(
        Int.(
            reverse(
                unsafe_wrap(
                    Vector{Cint},
                    Wrapper.mlx_array_shape(array.mlx_array),
                    Wrapper.mlx_array_ndim(array.mlx_array),
                ),
            )
        ),
    )
end

Base.IndexStyle(::Type{<:MLXArray}) = IndexLinear()

Base.getindex(array::MLXArray, i::Int) = getindex(unsafe_wrap(array), i)

function Base.setindex!(array::MLXArray{T, N}, v::T, i::Int) where {T, N}
    return setindex!(unsafe_wrap(array), v, i)
end

# StridedArray interface, cf. https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-strided-arrays

function Base.strides(array::MLXArray)
    return Tuple(
        Int.(
            reverse(
                unsafe_wrap(
                    Vector{Csize_t},
                    Wrapper.mlx_array_strides(array.mlx_array),
                    Wrapper.mlx_array_ndim(array.mlx_array),
                ),
            )
        ),
    )
end

function Base.unsafe_convert(::Type{Ptr{T}}, array::MLXArray{T, N}) where {T, N}
    mlx_array_data::Function = Wrapper.mlx_array_data_bool
    if T == Bool
        mlx_array_data = Wrapper.mlx_array_data_bool
    elseif T == UInt8
        mlx_array_data = Wrapper.mlx_array_data_uint8
    elseif T == UInt16
        mlx_array_data = Wrapper.mlx_array_data_uint16
    elseif T == UInt32
        mlx_array_data = Wrapper.mlx_array_data_uint32
    elseif T == UInt64
        mlx_array_data = Wrapper.mlx_array_data_uint64
    elseif T == Int8
        mlx_array_data = Wrapper.mlx_array_data_int8
    elseif T == Int16
        mlx_array_data = Wrapper.mlx_array_data_int16
    elseif T == Int32
        mlx_array_data = Wrapper.mlx_array_data_int32
    elseif T == Int64
        mlx_array_data = Wrapper.mlx_array_data_int64
        # TODO generate wrapper on system with HAS_FLOAT16
    elseif T == Float32
        mlx_array_data = Wrapper.mlx_array_data_float32
    elseif T == Float64
        mlx_array_data = Wrapper.mlx_array_data_float64
        # TODO generate wrapper on system with HAS_BFLOAT16
    elseif T == ComplexF32
        mlx_array_data = Wrapper.mlx_array_data_complex64
    else
        throw(ArgumentError("Unsupported type: $T"))
    end

    return mlx_array_data(array.mlx_array)
end

function Base.elsize(array::MLXArray{T, N}) where {T, N}
    return Int(Wrapper.mlx_array_itemsize(array.mlx_array))
end

function Base.unsafe_wrap(array::MLXArray{T, N}) where {T, N}
    return unsafe_wrap(Array, Base.unsafe_convert(Ptr{T}, array), size(array))
end
