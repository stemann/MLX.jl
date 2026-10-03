function Base.copy(a::MLXArray{T, N}) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    Wrapper.mlx_copy(result, a, s)
    return MLXArray{T, N}(result[])
end

"""
    dropdims(a::MLXArray; dims::Union{Dims, Integer, Nothing} = nothing)

Return an array with singleton dimensions removed. If `dims` is not specified,
all singleton dimensions are removed.
"""
function Base.dropdims(
    a::MLXArray{T, N}; dims::Union{Dims, Integer, Nothing} = nothing
) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    if isnothing(dims)
        Wrapper.mlx_squeeze_all(result, a, s)
    else
        if dims isa Integer
            dims = Dims(dims)
        end
        axes = collect(Cint.(dims) .- one(Cint))
        Wrapper.mlx_squeeze(result, a, axes, length(axes), s)
    end
    remaining_dims = Int(Wrapper.mlx_array_ndim(result[]))
    return MLXArray{T, remaining_dims}(result[])
end

function Base.sort(v::MLXVector{T}) where {T}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    Wrapper.mlx_sort_all(result, v, s)
    return MLXVector{T}(result[])
end

function Base.sort(a::MLXArray{T, N}; dims::Integer) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    axis = Cint(dims) - one(Cint)
    Wrapper.mlx_sort(result, a, axis, s)
    return MLXArray{T, N}(result[])
end

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

function elementwise(mlx_fn, a::MLXArray{T, N}, b::MLXArray{T}) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    mlx_fn(result, a, b, s)
    return MLXArray{T, N}(result[])
end

function Base.sortperm(v::MLXVector)
    zero_based_indices = astype(Int, argsort(v))
    return elementwise(Wrapper.mlx_add, zero_based_indices, MLXArray(fill(1)))
end

function Base.sortperm(a::MLXArray{T, N}; dims::Integer) where {T, N}
    # Linear indices, as for Array, from the indices along dims returned by argsort
    indices_along_dims = astype(Int, argsort(a; dims))
    n = size(a, dims)
    shape_along_dims = ntuple(d -> d == dims ? n : 1, N)
    positions_along_dims = MLXArray(reshape(collect(0:(n - 1)), shape_along_dims))
    stride_along_dims = MLXArray(fill(prod(size(a)[1:(dims - 1)])))
    offsets = elementwise(
        Wrapper.mlx_multiply,
        elementwise(Wrapper.mlx_subtract, indices_along_dims, positions_along_dims),
        stride_along_dims,
    )
    return elementwise(Wrapper.mlx_add, MLXArray(collect(LinearIndices(a))), offsets)
end

function Base.permutedims(a::MLXArray{T, N}, perm) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    axes = collect(Cint.(perm) .- one(Cint))
    Wrapper.mlx_transpose(result, a, axes, length(axes), s)
    # mlx_transpose yields a strided view; make it contiguous, like permutedims for Array
    transposed = MLXArray{T, N}(result[])
    result = Ref(Wrapper.mlx_array_new())
    Wrapper.mlx_contiguous(result, transposed, false, s)
    return MLXArray{T, N}(result[])
end

Base.permutedims(m::MLXMatrix) = permutedims(m, (2, 1))

for (fn, fn_def) in Private.get_unary_scalar_ops()
    @eval function Broadcast.broadcasted(
        ::Broadcast.ArrayStyle{MLXArray}, ::typeof($fn), a::MLXArray{T, N}
    ) where {T <: $(fn_def.TIn), N}
        s = get_stream()
        result_ref = Ref(Wrapper.mlx_array_new())
        $(fn_def.mlx_fn)(result_ref, a, s)
        result = MLXArray{$(fn_def.output_type)(T), N}(result_ref[])
        return N == 0 ? MLXNumber(result) : result # As materialize for 0-dim results
    end
end
