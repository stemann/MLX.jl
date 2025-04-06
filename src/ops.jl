"""
    dropdims(a::MLXArray; dims::Union{Dims, Integer, Nothing} = nothing)

Return an array with singleton dimensions removed. If `dims` is not specified,
all singleton dimensions are removed.
"""
function Base.dropdims(a::MLXArray{T, N}; dims::Union{Dims, Integer, Nothing} = nothing) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    if isnothing(dims)
        Wrapper.mlx_squeeze_all(result, a.mlx_array, s.mlx_stream)
    else
        if dims isa Integer
            dims = Dims(dims)
        end
        row_major_dims = ndims(a) .- dims .+ 1
        dims = collect(Cint.(row_major_dims) .- one(Cint))
        Wrapper.mlx_squeeze(result, a.mlx_array, dims, length(dims), s.mlx_stream)
    end
    remaining_dims = Int(Wrapper.mlx_array_ndim(result[]))
    return MLXArray{T, remaining_dims}(result[])
end

function Base.sort(v::MLXVector{T}) where {T}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    Wrapper.mlx_sort_all(result, v.mlx_array, s.mlx_stream)
    return MLXVector{T}(result[])
end

function Base.sort(a::MLXArray{T, N}; dims::Integer) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    row_major_dims = ndims(a) - dims + 1
    axis = Cint(row_major_dims) - one(Cint)
    Wrapper.mlx_sort(result, a.mlx_array, axis, s.mlx_stream)
    return MLXArray{T, N}(result[])
end

function Base.sortperm(v::MLXVector{T}) where {T}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    Wrapper.mlx_argsort_all(result, v.mlx_array, s.mlx_stream)
    return MLXVector{T}(result[])
end

function Base.sortperm(a::MLXArray{T, N}; dims::Integer) where {T, N}
    s = get_stream()
    result = Ref(Wrapper.mlx_array_new())
    row_major_dims = ndims(a) - dims + 1
    axis = Cint(row_major_dims) - one(Cint)
    Wrapper.mlx_argsort(result, a.mlx_array, axis, s.mlx_stream)
    return MLXArray{T, N}(result[])
end

for (fn, fn_def) in Private.get_unary_array_ops()
    TOut = fn_def.output_type(fn_def.TIn)
    kwargs_expr = Expr[]
    for (n, t) in pairs(fn_def.kwargs_types)
        push!(kwargs_expr, Expr(:(::), n, t))
    end

    @eval function Base.$fn(
        a::MLXArray{T, N}; $(kwargs_expr...)
    ) where {T <: $(fn_def.TIn), N}
        s = get_stream()
        result_ref = Ref(Wrapper.mlx_array_new())
        @static if isempty($(fn_def.kwargs_types))
            $(fn_def.mlx_fn)(result_ref, a.mlx_array, s.mlx_stream)
        else
        end
        @static if $(fn_def.preserves_type)
            return MLXArray{T, N}(result_ref[])
        else
            return MLXArray{$TOut, N}(result_ref[])
        end
    end
end

for (fn, fn_def) in Private.get_unary_scalar_ops()
    TOut = fn_def.output_type(fn_def.TIn)

    @eval function Broadcast.broadcasted(
        ::Broadcast.ArrayStyle{MLXArray}, ::typeof($fn), a::MLXArray{T, N}
    ) where {T <: $(fn_def.TIn), N}
        s = get_stream()
        result_ref = Ref(Wrapper.mlx_array_new())
        $(fn_def.mlx_fn)(result_ref, a.mlx_array, s.mlx_stream)
        @static if $(fn_def.preserves_type)
            return MLXArray{T, N}(result_ref[])
        else
            return MLXArray{$TOut, N}(result_ref[])
        end
    end
end
