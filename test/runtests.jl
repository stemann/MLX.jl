using Test

@testset verbose = true "MLX" begin
    include(joinpath(@__DIR__, "array_tests.jl"))
    include(joinpath(@__DIR__, "device_tests.jl"))
    include(joinpath(@__DIR__, "error_handling_tests.jl"))
    include(joinpath(@__DIR__, "metal_tests.jl"))
    if !Sys.iswindows() # Windows is hanging
        include(joinpath(@__DIR__, "stream_tests.jl"))
    end
    include(joinpath(@__DIR__, "string_tests.jl"))
end
