function metal_device_info()
    device_info = Wrapper.mlx_metal_device_info()
    return device_info
end

function metal_is_available()
    result = Ref(false)
    Wrapper.mlx_metal_is_available(result)
    return result[]
end
