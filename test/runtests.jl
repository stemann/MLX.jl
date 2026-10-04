using Test

@testset verbose = true "MLX" begin
    include(joinpath(@__DIR__, "ArrayTests.jl"))
    include(joinpath(@__DIR__, "DeviceTests.jl"))
    include(joinpath(@__DIR__, "ErrorHandlingTests.jl"))
    include(joinpath(@__DIR__, "NumberTests.jl"))
    include(joinpath(@__DIR__, "NumberTypesTests.jl"))
    include(joinpath(@__DIR__, "StreamTests.jl"))
end
