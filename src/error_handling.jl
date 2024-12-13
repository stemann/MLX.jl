struct MLXException <: Exception
    msg::String
end

function error_handler(s::Ptr{Cchar})::Cvoid
    throw(MLXException(unsafe_string(s)))
    return
end
