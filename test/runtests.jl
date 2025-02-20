using LinearAlgebra
using MLX
using Test

function run_tests()
    blas_config = BLAS.get_config()
    @info "Running tests using BLAS config: $blas_config"
    @testset verbose = true "MLX" begin
        include(joinpath(@__DIR__, "array_tests.jl"))
        include(joinpath(@__DIR__, "device_tests.jl"))
        include(joinpath(@__DIR__, "error_handling_tests.jl"))
        include(joinpath(@__DIR__, "stream_tests.jl"))
    end
end

blas_defined = haskey(ENV, "BLAS")
if blas_defined
    blas = ENV["BLAS"]
    @info "Ensuring selected BLAS is loaded: $blas"
    if blas == "AppleAccelerate"
        using AppleAccelerate
    elseif blas == "MKL"
        using MKL
    elseif blas == "OpenBLAS"
        # OpenBLAS should already be loaded
    end
end

run_tests()

if !blas_defined
    if Sys.ARCH == :x86_64 && (Sys.islinux() || Sys.iswindows())
        using MKL
        run_tests()
    elseif Sys.isapple()
        using AppleAccelerate
        run_tests()
    end
end
