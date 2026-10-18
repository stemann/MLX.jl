# Computed in `copy`, as `materialize` does, rather than in `broadcasted`, so that Base's
# `conj(A)`, `real(A)`, `imag(A)` and `-A`, which broadcast and keep 0-dim arrays, work too
for (fn, fn_def) in Internals.get_unary_ops()
    @eval function Base.copy(
        bc::Broadcast.Broadcasted{
            Broadcast.ArrayStyle{MLXArray}, <:Any, typeof($fn), <:Tuple{MLXArray{T, N}}
        },
    ) where {T <: $(fn_def.input_type), N}
        a = only(bc.args)
        s = get_stream()
        result_ref = Ref(Wrapper.mlx_array_new())
        $(fn_def.mlx_fn)(result_ref, a, s)
        return MLXArray{$(fn_def.output_type)(T), N}(result_ref[])
    end
end

# A broadcast over an `MLXNumber`'s 0-dim array, which gives an `MLXNumber`
for (fn, fn_def) in Internals.get_unary_ops()
    if fn in (:isfinite, :isinf, :isnan)
        # Predicates evaluate, as comparisons do, so they can be used for control flow
        @eval Base.$fn(x::MLXNumber{T}) where {T <: $(fn_def.input_type)} =
            Bool($fn.(x.array))
    else
        @eval Base.$fn(x::MLXNumber{T}) where {T <: $(fn_def.input_type)} = $fn.(x.array)
    end
end
