for (fn, fn_def) in Private.get_unary_ops()
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
