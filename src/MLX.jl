module MLX

export MLXException

include(joinpath(@__DIR__, "Wrapper.jl"))

include(joinpath(@__DIR__, "device.jl"))
include(joinpath(@__DIR__, "error_handling.jl"))

function __init__()
    register_error_handler()
    return nothing
end

end
