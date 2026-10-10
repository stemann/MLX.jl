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

function Base.sortperm(v::MLXVector{T}) where {T}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    Wrapper.mlx_argsort_all(result, v, s)
    return MLXVector{T}(result[])
end

function Base.sortperm(a::MLXArray{T, N}; dims::Integer) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    axis = Cint(dims) - one(Cint)
    Wrapper.mlx_argsort(result, a, axis, s)
    Wrapper.mlx_add(result, result[], Wrapper.mlx_array_new_int(1), s)
    return MLXArray{T, N}(result[])
end

function Base.permutedims(
    a::MLXArray{T, N}; dims::Union{Dims, Integer, Nothing} = nothing
) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    if isnothing(dims)
        Wrapper.mlx_transpose_all(result, a, s)
    else
        if dims isa Integer
            dims = Dims(dims)
        end
        axes = collect(Cint.(dims) .- one(Cint))
        Wrapper.mlx_transpose(result, a, axes, length(axes), s)
    end
    return MLXArray{T, N}(result[])
end

for (fn, fn_def) in Private.get_unary_scalar_ops()
    TOut = fn_def.output_type(fn_def.TIn) # TODO unused TOut; runtime inspection needed?

    @eval function Broadcast.broadcasted(
        ::Broadcast.ArrayStyle{MLXArray}, ::typeof($fn), a::MLXArray{T, N}
    ) where {T <: $(fn_def.TIn), N}
        s = get_stream()
        result_ref = Ref(Wrapper.mlx_array_new())
        $(fn_def.mlx_fn)(result_ref, a, s)
        @static if $(fn_def.preserves_type)
            return MLXArray{T, N}(result_ref[])
        else
            T_result = convert(Number, Wrapper.mlx_array_dtype(result_ref[])) # TODO unused TOut; runtime inspection needed?
            return MLXArray{T_result, N}(result_ref[])
        end
    end
end
