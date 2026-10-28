module MLXBFloat16sExt

using BFloat16s: BFloat16
using MLX: MLXArray, Wrapper

Base.convert(::Type{Wrapper.mlx_dtype}, ::Type{BFloat16}) = Wrapper.MLX_BFLOAT16

function Base.unsafe_convert(::Type{Ptr{BFloat16}}, array::MLXArray{BFloat16, N}) where {N}
    Wrapper.mlx_array_eval(array)
    return Ptr{BFloat16}(Wrapper.mlx_array_data_uint16(array)) # As for Float16
end

end
