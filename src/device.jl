struct Device
    mlx_device::Wrapper.mlx_device

    function Device()
        mlx_device = Wrapper.mlx_device_new()
        this = new(mlx_device)
        finalizer(d -> Wrapper.mlx_device_free(d.mlx_device), this)
        return this
    end
end

function get_default_device()
    device = Device()
    Wrapper.mlx_get_default_device(device.mlx_device)
    return device
end
