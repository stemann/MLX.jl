"""
    MLXNumber{T} <: Number

A scalar held by MLX as a 0-dimensional array, which, like an `MLXArray`, is evaluated lazily.

Comparisons and conversions to other `Number` types evaluate it.
"""
struct MLXNumber{T} <: Number
    array::MLXArray{T, 0}

    MLXNumber{T}(array::MLXArray{T, 0}) where {T} = new{T}(array)
end

# ccall keeps x.array alive and converts it with the unsafe_convert of MLXArray
Base.cconvert(::Type{Wrapper.mlx_array}, x::MLXNumber) = x.array

MLXNumber(array::MLXArray{T, 0}) where {T} = MLXNumber{T}(array)

# Number constructors

MLXNumber{T}(x::T) where {T <: Number} = MLXNumber{T}(MLXArray(fill(x)))
MLXNumber{T}(x::Number) where {T} = MLXNumber{T}(T(x))
# Both the above and the constructor for other Number types below match MLXNumber{T}(x::MLXNumber),
# neither more specifically, so the following method is needed to avoid an ambiguity
MLXNumber{T}(x::MLXNumber) where {T} = MLXNumber{T}(T(x))
MLXNumber{T}(x::MLXNumber{T}) where {T} = x
# MLX has scalar constructors for some types
for (T, mlx_fn) in (
    (Bool, :mlx_array_new_bool),
    (Int32, :mlx_array_new_int),
    (Float32, :mlx_array_new_float32),
    # (Float64, :mlx_array_new_float64), # TODO: mlx_array_new_float64 yields a float32 array in MLX C v0.1.2, cf. https://github.com/ml-explore/mlx-c/issues/58
)
    @eval MLXNumber{$T}(x::$T) = MLXNumber{$T}(MLXArray{$T, 0}(Wrapper.$mlx_fn(x)))
end
function MLXNumber{ComplexF32}(x::ComplexF32)
    return MLXNumber{ComplexF32}(
        MLXArray{ComplexF32, 0}(Wrapper.mlx_array_new_complex(real(x), imag(x)))
    )
end

MLXNumber(x::T) where {T <: Number} = MLXNumber{T}(x)
MLXNumber(x::MLXNumber) = x

# Constructor for other Number types, which also supports convert, cf.
# https://docs.julialang.org/en/v1/manual/conversion-and-promotion/#Defining-New-Conversions
(::Type{T})(x::MLXNumber) where {T <: Number} = T(x.array[])

# isequal and isless are defined separately, as Julia defines them to differ from == and < for NaN
# and -0.0
for op in (:(==), :<, :<=, :isequal, :isless)
    @eval begin
        Base.$op(a::MLXNumber, b::MLXNumber) = $op(a.array[], b.array[])
        Base.$op(a::MLXNumber, b::Number) = $op(a.array[], b)
        Base.$op(a::Number, b::MLXNumber) = $op(a, b.array[])
    end
end

Base.hash(x::MLXNumber, h::UInt) = hash(x.array[], h)

function Base.show(io::IO, x::MLXNumber)
    print(io, "MLXNumber(")
    show(io, x.array[])
    return print(io, ")")
end

# Broadcasting interface, as a 0-dimensional MLXArray

Base.broadcastable(x::MLXNumber) = x.array
