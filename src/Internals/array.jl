function is_available(array::Union{MLXArray, MLXNumber})
    available = Ref(false)
    Wrapper._mlx_array_is_available(available, array)
    return available[]
end

# The offset of the element at linear index i in MLX's data, which need not be row-major,
# e.g. after a transpose; the strides are only known once the array is evaluated, e.g. by
# unsafe_convert
function data_offset(array::MLXArray{T, N}, i::Int) where {T, N}
    index = CartesianIndices(array)[i]
    mlx_strides = Wrapper.mlx_array_strides(array)
    offsets = ntuple(d -> (index[d] - 1) * Int(unsafe_load(mlx_strides, d)), Val(N))
    return sum(offsets; init = 0)
end
