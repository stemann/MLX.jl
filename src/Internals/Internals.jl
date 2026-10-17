module Internals

using ..MLX:
    DeviceTypeCPU, DeviceTypeGPU, MLXArray, MLXVector, get_stream, supported_number_types
using ..Wrapper

include(joinpath(@__DIR__, "types.jl"))
include(joinpath(@__DIR__, "ops_array.jl"))
include(joinpath(@__DIR__, "testing.jl"))
include(joinpath(@__DIR__, "ops_unary.jl"))
include(joinpath(@__DIR__, "ops_binary.jl"))

end
