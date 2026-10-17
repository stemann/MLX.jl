module Internals

using ..MLX: DeviceTypeGPU, MLXArray, MLXNumber, MLXVector, get_stream
using ..Wrapper

include(joinpath(@__DIR__, "types.jl"))
include(joinpath(@__DIR__, "ops_array.jl"))
include(joinpath(@__DIR__, "testing.jl"))
include(joinpath(@__DIR__, "ops_unary.jl"))
include(joinpath(@__DIR__, "ops_binary.jl"))

end
