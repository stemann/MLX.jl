mutable struct MLXArray{T, N} <: AbstractArray{T, N}
    mlx_array::Wrapper.mlx_array

    function MLXArray{T, N}(mlx_array::Wrapper.mlx_array) where {T, N}
        this = new(mlx_array)
        finalizer(d -> Wrapper.mlx_array_free(d.mlx_array), this)
        return this
    end
end

# ccall keeps the result of cconvert (the MLXArray) alive during the call,
# so its finalizer cannot free the mlx_array that unsafe_convert passes to C
Base.cconvert(::Type{Wrapper.mlx_array}, array::MLXArray) = array
Base.unsafe_convert(::Type{Wrapper.mlx_array}, array::MLXArray) = array.mlx_array

function MLXArray{T, N}(array::AbstractArray{T, N}) where {T, N}
    is_column_major =
        storage_order(array; preferred_order = ArrayStorageOrderRow) ==
        ArrayStorageOrderColumn
    array_row_major = is_column_major ? permutedims(array, reverse(1:ndims(array))) : array
    shape = collect(Cint.(size(array)))
    dtype = convert(Wrapper.mlx_dtype, T)
    mlx_array = Wrapper.mlx_array_new_data(array_row_major, shape, Cint(N), dtype)
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

# UndefInitializer

function MLXArray{T, N}(::UndefInitializer, dims::Dims{N}) where {T, N}
    stream = get_stream()
    result_ref = Ref(Wrapper.mlx_array_new())
    shape = collect(Cint.(dims))
    dtype = convert(Wrapper.mlx_dtype, T)
    # MLX provides no uninitialized allocation, so undef arrays are zero-filled
    Wrapper.mlx_zeros(result_ref, shape, Cint(N), dtype, stream)
    return MLXArray{T, N}(result_ref[])
end

function MLXArray{T}(::UndefInitializer, dims::Dims{N}) where {T, N}
    return MLXArray{T, N}(undef, dims)
end

# BitArray

MLXArray{Bool, N}(array::BitArray{N}) where {N} = MLXArray(Array{Bool}(array))

MLXArray(array::BitArray{N}) where {N} = MLXArray{Bool, N}(array)

# AbstractArray interface, cf. https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

function Base.size(array::MLXArray{T, N}) where {T, N}
    return GC.@preserve array begin
        shape = Wrapper.mlx_array_shape(array)
        ntuple(d -> Int(unsafe_load(shape, d)), Val(N))
    end
end

Base.IndexStyle(::Type{<:MLXArray}) = IndexLinear()

function Base.getindex(array::MLXArray{T, N}, i::Int) where {T, N}
    @boundscheck checkbounds(array, i)
    return GC.@preserve array begin
        data = Base.unsafe_convert(Ptr{T}, array)
        unsafe_load(data, Internals.data_offset(array, i) + 1)
    end
end

function Base.setindex!(array::MLXArray{T, N}, v, i::Int) where {T, N}
    @boundscheck checkbounds(array, i)
    GC.@preserve array begin
        data = Base.unsafe_convert(Ptr{T}, array)
        unsafe_store!(data, convert(T, v), Internals.data_offset(array, i) + 1)
    end
    return array
end

function Base.similar(::MLXArray, ::Type{T}, dims::Dims{N}) where {T, N}
    return MLXArray{T, N}(undef, dims)
end

# Strided array interface, cf. https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-strided-arrays

function Base.strides(array::MLXArray{T, N}) where {T, N}
    array_strides = GC.@preserve array begin
        mlx_strides = Wrapper.mlx_array_strides(array)
        ntuple(d -> Int(unsafe_load(mlx_strides, d)), Val(N))
    end
    if any(iszero, array_strides) # Workaround for MLX issue where strides may be zero for dims of size 1: https://github.com/ml-explore/mlx/issues/2501
        non_zero_strides = map(s -> iszero(s) ? 1 : s, array_strides)
        @debug "Some strides are zero in $array_strides - returning strides $non_zero_strides for array of size $(size(array))"
        return non_zero_strides
    end

    return array_strides
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

    Wrapper.mlx_array_eval(array)
    return mlx_array_data(array)
end

Base.elsize(::Type{MLXArray{T, N}}) where {T, N} = sizeof(T)

function Base.elsize(array::MLXArray{T, N}) where {T, N}
    return Int(Wrapper.mlx_array_itemsize(array))
end

# Broadcasting interface, cf. https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting

Base.BroadcastStyle(::Type{<:MLXArray}) = Broadcast.ArrayStyle{MLXArray}()

function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MLXArray}}, ::Type{T}
) where {T}
    return similar(MLXArray{T}, axes(bc))
end

# Drop 0-dim results to scalars, as for `Array`, cf. https://github.com/JuliaLang/julia/issues/28866
function Base.Broadcast.materialize(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MLXArray}}
)
    result = copy(Broadcast.instantiate(bc))
    if iszero(ndims(result))
        return MLXNumber(result)
    end
    return result
end
