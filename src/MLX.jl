module MLX

export MLXException

include(joinpath(@__DIR__, "Wrapper.jl"))

include(joinpath(@__DIR__, "error_handling.jl"))

function __init__()
    error_handler_c = @cfunction(error_handler, Cvoid, (Ptr{Cchar},));
    Wrapper.mlx_set_error_handler(error_handler_c, C_NULL, C_NULL)
end

end
