function is_available(array::Union{MLXArray, MLXNumber})
    available = Ref(false)
    Wrapper._mlx_array_is_available(available, array)
    return available[]
end
