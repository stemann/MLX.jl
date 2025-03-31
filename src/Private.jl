module Private

using ..Wrapper

function return_input_type(::Type{TIn}) where {TIn}
    return TIn
end

function return_float_type(::Type{TIn}) where {TIn}
    return TIn <: Complex{<:AbstractFloat} ? TIn : Float32 # TODO: Float64 unsupported by MLX C 0.1.1
end

function get_unary_array_ops()
    return Dict(
        :sortperm => (
            mlx_fn = Wrapper.mlx_argsort_all, # TODO check if this is correct # mlx_argsort
            TIn = Number,
            output_type = return_input_type,
            preserves_type = true,
            kwargs_types = (; dims = Integer),
            normalize = (a, TIn) -> a,
        ),
        :copy => (
            mlx_fn = Wrapper.mlx_copy,
            TIn = Number,
            output_type = return_input_type,
            preserves_type = true,
            kwargs_types = NamedTuple(),
            normalize = (a, TIn) -> a,
        ),
        :sort => (
            mlx_fn = Wrapper.mlx_sort_all, # TODO check if this is correct # mlx_sort
            TIn = Number,
            output_type = return_input_type,
            preserves_type = true,
            kwargs_types = (; dims = Integer),
            normalize = (a, TIn) -> a,
        ),
        :dropdims => (
            mlx_fn = Wrapper.mlx_squeeze_all, # mlx_squeeze
            TIn = Number,
            output_type = return_input_type,
            preserves_type = true,
            kwargs_types = (; dims = Integer),
            normalize = (a, TIn) -> a,
        ),
        :transpose => ( # TODO: testing fails for transpose.(::MLXArray{Bool, 2}), (2, 1) likely due to array storage order. Bool[0 1] == Bool[0; 1;;]
            mlx_fn = Wrapper.mlx_transpose_all, # mlx_transpose
            TIn = Number,
            output_type = return_input_type,
            preserves_type = true,
            kwargs_types = NamedTuple(),
            normalize = (a, TIn) -> a,
        ),
    )
end

function get_unary_scalar_ops()
    RealExceptBool = Union{AbstractFloat, Signed, Unsigned}
    return Dict(
        :abs => (
            mlx_fn = Wrapper.mlx_abs,
            TIn = Real, # in testing, abs differs from mlx_abs wrt. Complex{<:AbstractFloat}
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :acos => (
            mlx_fn = Wrapper.mlx_arccos,
            TIn = RealExceptBool, # in testing, acos differs from mlx_arccos wrt. Bool, Complex{<:AbstractFloat}
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> TIn.(floor.(a ./ maximum(a))),
        ),
        :acosh => (
            mlx_fn = Wrapper.mlx_arccosh,
            TIn = Union{AbstractFloat, Complex}, # in testing, acosh differs from mlx_arccosh wrt. Integer
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a .+ 1,
        ),
        :asin => (
            mlx_fn = Wrapper.mlx_arcsin,
            TIn = AbstractFloat, # in testing, asin differs from mlx_arcsin wrt. Integer, normalize fails for Complex{<:AbstractFloat}
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> TIn.(floor.(a ./ maximum(a))),
        ),
        :asinh => (
            mlx_fn = Wrapper.mlx_arcsinh,
            TIn = Union{AbstractFloat, Complex}, # in testing, asinh differs from mlx_arcsinh wrt. Integer
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :atan => (
            mlx_fn = Wrapper.mlx_arctan,
            TIn = Real, # testing fails for atan wrt. Complex{<:AbstractFloat}
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        :atanh => (
            mlx_fn = Wrapper.mlx_arctanh,
            TIn = AbstractFloat, # in testing, atanh differs from mlx_arctanh wrt. Integer, normalize fails for Complex{<:AbstractFloat}
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> TIn.(floor.(a ./ maximum(a))),
        ),
        # mlx_atleast_1d
        # mlx_atleast_2d
        # mlx_atleast_3d
        :~ => (
            mlx_fn = Wrapper.mlx_bitwise_invert,
            TIn = Integer,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :ceil => (
            mlx_fn = Wrapper.mlx_ceil,
            TIn = Real, # MLX: [floor] Not supported for complex64
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :conj => ( # TODO: conj is also defined for AbstractArray
            mlx_fn = Wrapper.mlx_conjugate,
            TIn = Number,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :cos => (
            mlx_fn = Wrapper.mlx_cos,
            TIn = AbstractFloat, # in testing, cos differs from mlx_cos wrt. Signed, Unsigned, Complex{<:AbstractFloat}, Bool fails: conversion to pointer not defined for BitArray
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) ->
                TIn.(round.(map(x -> iszero(x % π) ? x + eps(Float32) : x, a))),
        ),
        :cosh => (
            mlx_fn = Wrapper.mlx_cosh,
            TIn = Real, # testing fails for cosh wrt. Complex{<:AbstractFloat}
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        :rad2deg => (
            mlx_fn = Wrapper.mlx_degrees,
            TIn = Real, # testing fails for rad2deg wrt. Complex{<:AbstractFloat}
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        # mlx_erf
        # mlx_erfinv
        :exp => (
            mlx_fn = Wrapper.mlx_exp,
            TIn = Real, # testing fails for exp wrt. Complex{<:AbstractFloat}. TODO: Needs broadcast across Float32 and ComplexF32: `copyto!(dest::MLXArray{Float32, 3}, bc::Base.Broadcast.Broadcasted{Nothing, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}, Base.OneTo{Int64}}, typeof(-), Tuple{MLXArray{Float32, 3}, Array{ComplexF32, 3}}})`
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        :expm1 => (
            mlx_fn = Wrapper.mlx_expm1,
            TIn = Real, # testing fails for expm1 wrt. Complex{<:AbstractFloat}. TODO: Needs broadcast across Float32 and ComplexF32
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        :floor => (
            mlx_fn = Wrapper.mlx_floor,
            TIn = Real, # MLX: [floor] Not supported for complex64
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :imag => (
            mlx_fn = Wrapper.mlx_imag,
            TIn = Real, # testing segfaults wrt. Complex{<:AbstractFloat}
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :isfinite => (
            mlx_fn = Wrapper.mlx_isfinite,
            TIn = Number,
            output_type = (::Type) -> Bool,
            preserves_type = false,
            normalize = (a, TIn) -> a, # TODO should provide some finite/Infs
        ),
        :isinf => (
            mlx_fn = Wrapper.mlx_isinf,
            TIn = Number,
            output_type = (::Type) -> Bool,
            preserves_type = false,
            normalize = (a, TIn) -> a, # TODO should provide some Infs
        ),
        :isnan => (
            mlx_fn = Wrapper.mlx_isnan,
            TIn = Number,
            output_type = (::Type) -> Bool,
            preserves_type = false,
            normalize = (a, TIn) -> a, # TODO should provide some NaNs
        ),
        # mlx_isneginf
        # mlx_isposinf
        :log => (
            mlx_fn = Wrapper.mlx_log,
            TIn = RealExceptBool, # Bool fails: conversion to pointer not defined for BitArray. Complex{<:AbstractFloat} fails: MethodError: no method matching isless(::ComplexF32, ::Float32)
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> TIn.(ceil.(max.(eps(Float32), a))),
        ),
        :log10 => (
            mlx_fn = Wrapper.mlx_log10,
            TIn = RealExceptBool, # Bool fails: conversion to pointer not defined for BitArray. Complex{<:AbstractFloat} fails: MethodError: no method matching isless(::ComplexF32, ::Float32)
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> TIn.(ceil.(max.(eps(Float32), a))),
        ),
        :log1p => (
            mlx_fn = Wrapper.mlx_log1p,
            TIn = RealExceptBool, # Bool fails: conversion to pointer not defined for BitArray. Complex{<:AbstractFloat} fails: MethodError: no method matching isless(::ComplexF32, ::Float32)
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> TIn.(ceil.(max.(eps(Float32), a))),
        ),
        :log2 => (
            mlx_fn = Wrapper.mlx_log2,
            TIn = RealExceptBool, # Bool fails: conversion to pointer not defined for BitArray. Complex{<:AbstractFloat} fails: MethodError: no method matching isless(::ComplexF32, ::Float32)
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> TIn.(ceil.(max.(eps(Float32), a))),
        ),
        :! => (
            mlx_fn = Wrapper.mlx_logical_not,
            TIn = Bool,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :- => (
            mlx_fn = Wrapper.mlx_negative,
            TIn = Union{RealExceptBool, Complex{<:AbstractFloat}}, # MLX: [negative] Not supported for bool, use logical_not instead.
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        # mlx_ones_like
        :deg2rad => (
            mlx_fn = Wrapper.mlx_radians,
            TIn = Real, # testing fails for deg2rad wrt. Complex{<:AbstractFloat}. TODO: Needs broadcast across Float32 and ComplexF32
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        :real => (
            mlx_fn = Wrapper.mlx_real,
            TIn = Real, # testing fails for real wrt. Complex{<:AbstractFloat} likely due to array storage order.
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :inv => (
            mlx_fn = Wrapper.mlx_reciprocal, # TODO check if this is correct, notably wrt. mlx_linalg_inv
            TIn = Real, # testing fails for inv wrt. Complex{<:AbstractFloat}. TODO: Needs broadcast across Float32 and ComplexF32
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        # mlx_rsqrt
        # mlx_sigmoid
        :sign => (
            mlx_fn = Wrapper.mlx_sign,
            TIn = Union{AbstractFloat, Bool, Signed, Complex}, # TIn = Number \ Unsigned: sign broken on CPU for Unsigned on MLX <= 0.24.1, cf. https://github.com/ml-explore/mlx/issues/2023
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :sin => (
            mlx_fn = Wrapper.mlx_sin,
            TIn = AbstractFloat, # in testing, sin differs from mlx_sin wrt. Signed, Unsigned, Complex{<:AbstractFloat}, Bool fails: conversion to pointer not defined for BitArray
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) ->
                TIn.(round.(map(x -> iszero(x % π / 2) ? x + eps(Float32) : x, a))),
        ),
        :sinh => (
            mlx_fn = Wrapper.mlx_sinh,
            TIn = Real, # testing fails for cosh wrt. Complex{<:AbstractFloat}. TODO: Needs broadcast across Float32 and ComplexF32
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        :sqrt => (
            mlx_fn = Wrapper.mlx_sqrt,
            TIn = RealExceptBool, # Bool fails: conversion to pointer not defined for BitArray. Complex{<:AbstractFloat} fails: MethodError: no method matching isless(::ComplexF32, ::Float32)
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> TIn.(ceil.(max.(eps(Float32), a))),
        ),
        # mlx_square
        # mlx_stop_gradient
        :tan => (
            mlx_fn = Wrapper.mlx_tan,
            TIn = Union{AbstractFloat, Bool}, # in testing, tan differs from mlx_tan wrt. Signed, Unsigned, Complex{<:AbstractFloat}
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        :tanh => (
            mlx_fn = Wrapper.mlx_tanh,
            TIn = Real, # testing fails for tanh wrt. Complex{<:AbstractFloat}. TODO: Needs broadcast across Float32 and ComplexF32
            output_type = return_float_type,
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        # mlx_linalg_inv
        # :pinv => ( # TODO using LinearAlgebra
        #     mlx_fn = Wrapper.mlx_linalg_pinv,
        #     TIn = Number,
        #     output_type = return_input_type,
        #     preserves_type = true,
        #     normalize = (a, TIn) -> a,
        # ),
    )
end

end
