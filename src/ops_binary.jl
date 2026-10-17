# Binary ops convert both operands to Julia's result type, or the op's compute type, as MLX
# promotes types differently, e.g. Bool + Bool gives Bool
for (fn, fn_def) in Internals.get_binary_ops()
    compute_type = if isnothing(fn_def.compute_type)
        :(Base.promote_op($fn, T, S))
    else
        :($(fn_def.compute_type)(T, S))
    end
    @eval function Base.copy(
        bc::Broadcast.Broadcasted{
            Broadcast.ArrayStyle{MLXArray},
            <:Any,
            typeof($fn),
            <:Tuple{Union{MLXArray{T}, T}, Union{MLXArray{S}, S}},
        },
    ) where {T <: $(fn_def.input_type), S <: $(fn_def.input_type)}
        a, b = bc.args
        C = $compute_type
        N = max(ndims(a), ndims(b))
        a = Internals.broadcast_operand(C, a, Val(N))
        b = Internals.broadcast_operand(C, b, Val(N))
        s = get_stream()
        result = Ref(Wrapper.mlx_array_new())
        $(fn_def.mlx_fn)(result, a, b, s)
        return MLXArray{C, N}(result[])
    end
end
