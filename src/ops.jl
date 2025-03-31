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
