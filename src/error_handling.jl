struct MLXException <: Exception
    msg::String
end

function error_handler(s::Ptr{Cchar})::Cvoid
    throw(MLXException(unsafe_string(s)))
    return nothing
end

function register_error_handler()
    error_handler_c = @cfunction(error_handler, Cvoid, (Ptr{Cchar},))
    Wrapper.mlx_set_error_handler(error_handler_c, C_NULL, C_NULL)
    return nothing
end
