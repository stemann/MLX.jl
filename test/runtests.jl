using Test

@testset "MLX" begin
    include(joinpath(@__DIR__, "error_handling_tests.jl"))
end
