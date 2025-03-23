module MLX

@static if VERSION < v"1.11"
    using ScopedValues
else
    using Base.ScopedValues
end

export MLXArray, MLXException, MLXMatrix, MLXVecOrMat, MLXVector

include(joinpath(@__DIR__, "Wrapper.jl"))

include(joinpath(@__DIR__, "Private.jl"))

include(joinpath(@__DIR__, "array.jl"))
include(joinpath(@__DIR__, "device.jl"))
include(joinpath(@__DIR__, "error_handling.jl"))
include(joinpath(@__DIR__, "metal.jl"))
include(joinpath(@__DIR__, "stream.jl"))

include(joinpath(@__DIR__, "ops.jl"))

function __init__()
    register_error_handler()
    return nothing
end

end
