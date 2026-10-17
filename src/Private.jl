module Private

using ..MLX: MLXArray, MLXVector, get_stream
using ..Wrapper

function argsort(a::MLXArray{<:Any, N}; dims::Union{Integer, Nothing} = nothing) where {N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    if isnothing(dims)
        Wrapper.mlx_argsort_all(result, a, s)
        return MLXVector{UInt32}(result[])
    end
    Wrapper.mlx_argsort(result, a, Cint(dims) - one(Cint), s)
    return MLXArray{UInt32, N}(result[])
end

function astype(::Type{T}, a::MLXArray{<:Any, N}) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    dtype = convert(Wrapper.mlx_dtype, T)
    Wrapper.mlx_astype(result, a, dtype, s)
    return MLXArray{T, N}(result[])
end

function binary_op(mlx_fn, a::MLXArray{T, N}, b::MLXArray{T, M}) where {T, N, M}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    mlx_fn(result, a, b, s)
    return MLXArray{T, max(N, M)}(result[])
end

end
