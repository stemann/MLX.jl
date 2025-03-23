for (fn, mlx_fn) in Private.unary_ops
    @eval function Broadcast.broadcasted(::Broadcast.ArrayStyle{MLXArray}, ::typeof($fn), a::MLXArray{T, N}) where {T, N}
        s = get_stream()
        result_ref = Ref(Wrapper.mlx_array_new())
        $mlx_fn(result_ref, a.mlx_array, s.mlx_stream)
        return MLXArray{T, N}(result_ref[]) # TODO result_ref might not be of type MLXArray{T, N}
    end
end
