function metal_is_available()
    result = Ref(false)
    Wrapper.mlx_metal_is_available(result)
    return result[]
end
