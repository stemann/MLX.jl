mutable struct MLXArray{T, N} <: AbstractArray{T, N}
    mlx_array::Wrapper.mlx_array

    function MLXArray{T, N}(mlx_array::Wrapper.mlx_array) where {T, N}
        this = new(mlx_array)
        finalizer(d -> Wrapper.mlx_array_free(d.mlx_array), this)
        return this
    end
end

function MLXArray{T, N}(array::AbstractArray{T, N}) where {T, N}
    is_column_major =
        storage_order(array; preferred_order = ArrayStorageOrderRow) ==
        ArrayStorageOrderColumn
    array_row_major = is_column_major ? permutedims(array, reverse(1:ndims(array))) : array
    shape = collect(Cint.(size(array)))
    dtype = convert(Wrapper.mlx_dtype, T)
    mlx_array = GC.@preserve array_row_major shape Wrapper.mlx_array_new_data(
        pointer(array_row_major), pointer(shape), Cint(N), dtype
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

# UndefInitializer

function MLXArray{T, N}(::UndefInitializer, dims::Dims{N}) where {T, N}
    stream = get_stream()
    result_ref = Ref(Wrapper.mlx_array_new())
    shape = collect(Cint.(dims))
    dtype = convert(Wrapper.mlx_dtype, T)
    Wrapper.mlx_zeros(result_ref, pointer(shape), Cint(N), dtype, stream.mlx_stream)
    return MLXArray{T, N}(result_ref[])
end

function MLXArray{T}(::UndefInitializer, dims::Dims{N}) where {T, N}
    return MLXArray{T, N}(undef, dims)
end

# BitArray

MLXArray{Bool, N}(array::BitArray{N}) where {N} = MLXArray(Array{Bool}(array))

MLXArray(array::BitArray{N}) where {N} = MLXArray{Bool, N}(array)

# AbstractArray interface, cf. https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

function Base.size(array::MLXArray)
    return Tuple(
        Int.(
            unsafe_wrap(
                Vector{Cint},
                Wrapper.mlx_array_shape(array.mlx_array),
                Wrapper.mlx_array_ndim(array.mlx_array),
            ),
        ),
    )
end

Base.IndexStyle(::Type{<:MLXArray}) = IndexLinear()

Base.getindex(array::MLXArray, i::Int) = getindex(unsafe_wrap(array), i)

function Base.setindex!(array::MLXArray{T, N}, v::T, i::Int) where {T, N}
    setindex!(unsafe_wrap(array), v, i)
    return array
end

function Base.similar(array::MLXArray{T, N}, ::Type{T}, ::Dims{N}) where {T, N}
    stream = get_stream()
    result_ref = Ref(Wrapper.mlx_array_new())
    Wrapper.mlx_zeros_like(result_ref, array.mlx_array, stream.mlx_stream)
    return MLXArray{T, N}(result_ref[])
end

# Strided array interface, cf. https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-strided-arrays

function Base.strides(array::MLXArray)
    array_strides = Tuple(
        Int.(
            unsafe_wrap(
                Vector{Csize_t},
                Wrapper.mlx_array_strides(array.mlx_array),
                Wrapper.mlx_array_ndim(array.mlx_array),
            ),
        ),
    )
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

    Wrapper.mlx_array_eval(array.mlx_array)
    return mlx_array_data(array.mlx_array)
end

Base.elsize(::Type{MLXArray{T, N}}) where {T, N} = sizeof(T)

function Base.elsize(array::MLXArray{T, N}) where {T, N}
    return Int(Wrapper.mlx_array_itemsize(array.mlx_array))
end

function Base.unsafe_wrap(array::MLXArray{T, N}) where {T, N}
    is_column_major = storage_order(array) == ArrayStorageOrderColumn
    size_column_major = is_column_major ? size(array) : reverse(size(array))
    wrapped_array = unsafe_wrap(
        Array, Base.unsafe_convert(Ptr{T}, array), size_column_major
    )
    if is_column_major
        return PermutedDimsArray(wrapped_array, 1:ndims(array))
    else
        return PermutedDimsArray(wrapped_array, reverse(1:ndims(array)))
    end
end

# Broadcasting interface, cf. https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting

Base.BroadcastStyle(::Type{<:MLXArray}) = Broadcast.ArrayStyle{MLXArray}()

function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MLXArray}}, ::Type{TElement}
) where {TElement}
    first_mlx_array(bc::Broadcast.Broadcasted) = first_mlx_array(bc.args)
    function first_mlx_array(args::Tuple)
        return first_mlx_array(first_mlx_array(args[1]), Base.tail(args))
    end
    first_mlx_array(x) = x
    first_mlx_array(::Tuple{}) = nothing
    first_mlx_array(a::MLXArray, _) = a
    first_mlx_array(::Any, rest) = first_mlx_array(rest)
    mlx_array = first_mlx_array(bc)
    if isnothing(mlx_array)
        return similar(MLXArray{TElement}, ())
    end
    return similar(mlx_array)
end

function Base.Broadcast.materialize(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MLXArray}}
)
    result = copy(Broadcast.instantiate(bc))
    if iszero(ndims(result)) # Drop 0-dim arrays to scalars, cf. https://github.com/JuliaLang/julia/issues/28866
        return result[]
    end
    return result
end
