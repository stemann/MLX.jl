module NumberTests

using MLX
using Random
using Test

include(joinpath(@__DIR__, "TestUtils.jl"))
using .TestUtils

@testset "MLXNumber" begin
    Random.seed!(42)

    rand_value(T) =
        rand(T <: Integer && T != Bool ? (T <: Signed ? (T(-10):T(10)) : (T(0):T(10))) : T)

    mlx_dtype(x::MLXNumber) = convert(Number, MLX.Wrapper.mlx_array_dtype(x))

    @test MLXNumber <: Number
    @test repr(MLXNumber(2.0f0)) == "MLXNumber(2.0f0)"

    testset_foreach_type("Constructors") do T
        value = rand_value(T)
        x = MLXNumber(value)
        @test x isa MLXNumber{T}
        @test mlx_dtype(x) == T
        @test MLXNumber(x.array) isa MLXNumber{T}
        @test MLXNumber(x) === x
        @test MLXNumber{T}(x) === x
        @test MLXNumber{T}(value) isa MLXNumber{T}
        # Another MLXNumber type, to which all supported types convert
        if T != ComplexF32
            y = MLXNumber{ComplexF32}(x)
            @test y isa MLXNumber{ComplexF32}
            @test mlx_dtype(y) == ComplexF32
            @test ComplexF32(y) == ComplexF32(value)
        end
    end

    testset_foreach_type("Conversions") do T
        value = rand_value(T)
        x = MLXNumber(value)
        @test convert(T, x) === value
        @test T(x) === value
        @test convert(MLXNumber{T}, value) isa MLXNumber{T}
    end

    testset_foreach_type("zero and one") do T
        x = MLXNumber(rand_value(T))
        @test zero(x) isa MLXNumber{T}
        @test T(zero(x)) == zero(T)
        @test T(one(x)) == one(T)
    end

    testset_foreach_type("Equality and hashing") do T
        a = rand_value(T)
        b = rand_value(T)
        @test (MLXNumber(a) == MLXNumber(b)) === (a == b)
        @test (MLXNumber(a) == b) === (a == b)
        @test (a == MLXNumber(b)) === (a == b)
        @test isequal(MLXNumber(a), a)
        @test hash(MLXNumber(a)) == hash(a)
    end

    real_types(device_type) =
        filter(T -> T <: Real, MLX.supported_number_types(device_type))
    testset_foreach_type("Order"; element_types = real_types) do T
        a = rand_value(T)
        b = rand_value(T)
        for op in (<, <=, >, >=, isless)
            @test op(MLXNumber(a), MLXNumber(b)) === op(a, b)
            @test op(MLXNumber(a), b) === op(a, b)
            @test op(a, MLXNumber(b)) === op(a, b)
        end
    end

    @testset "Comparisons with NaN" begin
        @test MLXNumber(NaN32) != MLXNumber(NaN32)
        @test isequal(MLXNumber(NaN32), MLXNumber(NaN32))
        @test !(MLXNumber(NaN32) < 1)
    end
end

end
