using Test

@testset "MLX" begin
    include(joinpath(@__DIR__, "device_tests.jl"))
    include(joinpath(@__DIR__, "metal_tests.jl"))
end
