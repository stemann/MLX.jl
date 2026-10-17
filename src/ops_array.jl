function Base.copy(a::MLXArray{T, N}) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    Wrapper.mlx_copy(result, a, s)
    return MLXArray{T, N}(result[])
end

function Base.dropdims(a::MLXArray{T, N}; dims::Union{Dims, Integer}) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    axes = collect(Cint.(dims isa Integer ? (dims,) : dims) .- one(Cint))
    Wrapper.mlx_squeeze(result, a, axes, length(axes), s)
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

function Base.sortperm(v::MLXVector)
    zero_based_indices = Internals.astype(Int, Internals.argsort(v))
    return zero_based_indices .+ 1
end

function Base.sortperm(a::MLXArray{T, N}; dims::Integer) where {T, N}
    # Linear indices, as for Array, from the indices along dims returned by argsort
    perm_indices_along_dims = Internals.astype(Int, Internals.argsort(a; dims))
    n = size(a, dims)
    shape_along_dims = ntuple(d -> d == dims ? n : 1, N)
    positions_along_dims = MLXArray(reshape(collect(0:(n - 1)), shape_along_dims))
    stride_along_dims = size_to_strides_column(size(a))[dims]
    perm_steps_along_dims = perm_indices_along_dims .- positions_along_dims
    perm_linear_offsets = perm_steps_along_dims .* stride_along_dims
    return MLXArray(collect(LinearIndices(a))) .+ perm_linear_offsets
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
