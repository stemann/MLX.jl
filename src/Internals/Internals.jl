module Internals

using ..MLX: DeviceTypeGPU, MLXArray, MLXVector, get_stream
using ..Wrapper

include(joinpath(@__DIR__, "types.jl"))
include(joinpath(@__DIR__, "ops_array.jl"))
include(joinpath(@__DIR__, "testing.jl"))
include(joinpath(@__DIR__, "ops_unary.jl"))

end
