function argsort(a::MLXArray{<:Any, N}; dims::Union{Integer, Nothing} = nothing) where {N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    if isnothing(dims)
        Wrapper.mlx_argsort(result, a, s)
        return MLXVector{UInt32}(result[])
    end
    Wrapper.mlx_argsort_axis(result, a, Cint(dims) - one(Cint), s)
    return MLXArray{UInt32, N}(result[])
end

function astype(::Type{T}, a::MLXArray{<:Any, N}) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    dtype = convert(Wrapper.mlx_dtype, T)
    Wrapper.mlx_astype(result, a, dtype, s)
    return MLXArray{T, N}(result[])
end

# MLX's reshape, which orders elements row-major, unlike Base's
function reshape(a::MLXArray{T}, dims::Dims{N}) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    shape = Cint[dims...]
    Wrapper.mlx_reshape(result, a, shape, length(shape), s)
    return MLXArray{T, N}(result[])
end
