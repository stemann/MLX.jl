module MLX

include(joinpath(@__DIR__, "Wrapper.jl"))

include(joinpath(@__DIR__, "device.jl"))
include(joinpath(@__DIR__, "metal.jl"))

end
