module Private

using ..Wrapper

function return_input_type(::Type{TIn}) where {TIn}
    return TIn
end

function return_float_type(::Type{TIn}) where {TIn}
    return TIn <: Union{AbstractFloat, Complex{<:AbstractFloat}} ? TIn : Float32
end

function return_real_type(::Type{TIn}) where {TIn}
    return TIn <: Complex{<:AbstractFloat} ? real(TIn) : TIn
end

# Replaces leading elements of floating-point arrays by Inf, -Inf and NaN, for testing
function with_nonfinite_values(a, ::Type{TIn}) where {TIn}
    TIn <: Union{AbstractFloat, Complex{<:AbstractFloat}} || return a
    b = copy(a)
    for (i, value) in zip(eachindex(b), TIn[Inf, -Inf, NaN])
        b[i] = value
    end
    return b
end

function get_unary_scalar_ops()
    RealExceptBool = Union{AbstractFloat, Signed, Unsigned}
    return Dict(
        :abs => (
            mlx_fn = Wrapper.mlx_abs,
            TIn = Number,
            output_type = return_real_type,
            normalize = (a, TIn) -> a,
        ),
        :acos => (
            mlx_fn = Wrapper.mlx_arccos,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> TIn <: Real ? TIn.(clamp.(a, -1, 1)) : a,
        ),
        :acosh => (
            mlx_fn = Wrapper.mlx_arccosh,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> begin
                TIn <: Complex && return a .+ 1
                TIn == Bool && return trues(size(a)) # acosh(false) is undefined
                return TIn.(abs.(a) .+ 1)
            end,
        ),
        :asin => (
            mlx_fn = Wrapper.mlx_arcsin,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> TIn <: Real ? TIn.(clamp.(a, -1, 1)) : a,
        ),
        :asinh => (
            mlx_fn = Wrapper.mlx_arcsinh,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        :atan => (
            mlx_fn = Wrapper.mlx_arctan,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        :atanh => (
            mlx_fn = Wrapper.mlx_arctanh,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> TIn <: Real ? TIn.(clamp.(a, -1, 1)) : a,
        ),
        # mlx_atleast_1d
        # mlx_atleast_2d
        # mlx_atleast_3d
        :~ => (
            mlx_fn = Wrapper.mlx_bitwise_invert,
            TIn = Integer,
            output_type = return_input_type,
            normalize = (a, TIn) -> a,
        ),
        :ceil => (
            mlx_fn = Wrapper.mlx_ceil,
            TIn = Real, # MLX: [floor] Not supported for complex64
            output_type = return_input_type,
            normalize = (a, TIn) -> a,
        ),
        :conj => ( # conj(::AbstractArray) broadcasts conj, so it also uses mlx_conjugate
            mlx_fn = Wrapper.mlx_conjugate,
            TIn = Number,
            output_type = return_input_type,
            normalize = (a, TIn) -> a,
        ),
        :cos => (
            mlx_fn = Wrapper.mlx_cos,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> begin
                TIn <: Complex && return a
                return TIn.(round.(map(x -> iszero(x % π) ? x + eps(Float32) : x, a)))
            end,
        ),
        :cosh => (
            mlx_fn = Wrapper.mlx_cosh,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        :rad2deg => (
            mlx_fn = Wrapper.mlx_degrees,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        # mlx_erf
        # mlx_erfinv
        :exp => (
            mlx_fn = Wrapper.mlx_exp,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        :expm1 => (
            mlx_fn = Wrapper.mlx_expm1,
            TIn = Real, # MLX: expm1 of complex inputs ignores the imaginary part
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        :floor => (
            mlx_fn = Wrapper.mlx_floor,
            TIn = Real, # MLX: [floor] Not supported for complex64
            output_type = return_input_type,
            normalize = (a, TIn) -> a,
        ),
        :imag => (
            mlx_fn = Wrapper.mlx_imag,
            TIn = Number,
            output_type = return_real_type,
            normalize = (a, TIn) -> a,
        ),
        :isfinite => (
            mlx_fn = Wrapper.mlx_isfinite,
            TIn = Real, # MLX: isfinite(complex(0, Inf)) is true
            output_type = (::Type) -> Bool,
            normalize = with_nonfinite_values,
        ),
        :isinf => (
            mlx_fn = Wrapper.mlx_isinf,
            TIn = Real, # MLX: isinf(complex(0, Inf)) is false
            output_type = (::Type) -> Bool,
            normalize = with_nonfinite_values,
        ),
        :isnan => (
            mlx_fn = Wrapper.mlx_isnan,
            TIn = Number,
            output_type = (::Type) -> Bool,
            normalize = with_nonfinite_values,
        ),
        # mlx_isneginf
        # mlx_isposinf
        :log => (
            mlx_fn = Wrapper.mlx_log,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) ->
                TIn <: Real ? TIn.(ceil.(max.(eps(Float32), a))) : a,
        ),
        :log10 => (
            mlx_fn = Wrapper.mlx_log10,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) ->
                TIn <: Real ? TIn.(ceil.(max.(eps(Float32), a))) : a,
        ),
        :log1p => (
            mlx_fn = Wrapper.mlx_log1p,
            TIn = Real, # MLX: log1p of complex inputs ignores the imaginary part
            output_type = return_float_type,
            normalize = (a, TIn) ->
                TIn <: Real ? TIn.(ceil.(max.(eps(Float32), a))) : a,
        ),
        :log2 => (
            mlx_fn = Wrapper.mlx_log2,
            TIn = Real, # MLX: log2 of complex inputs ignores the imaginary part
            output_type = return_float_type,
            normalize = (a, TIn) ->
                TIn <: Real ? TIn.(ceil.(max.(eps(Float32), a))) : a,
        ),
        :! => (
            mlx_fn = Wrapper.mlx_logical_not,
            TIn = Bool,
            output_type = return_input_type,
            normalize = (a, TIn) -> a,
        ),
        :- => (
            mlx_fn = Wrapper.mlx_negative,
            TIn = Union{RealExceptBool, Complex{<:AbstractFloat}}, # MLX: [negative] Not supported for bool, use logical_not instead.
            output_type = return_input_type,
            normalize = (a, TIn) -> a,
        ),
        # mlx_ones_like
        :deg2rad => (
            mlx_fn = Wrapper.mlx_radians,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        :real => (
            mlx_fn = Wrapper.mlx_real,
            TIn = Number,
            output_type = return_real_type,
            normalize = (a, TIn) -> a,
        ),
        :inv => (
            mlx_fn = Wrapper.mlx_reciprocal, # Elementwise; mlx_linalg_inv is the matrix inverse
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        # mlx_rsqrt
        # mlx_sigmoid
        :sign => (
            mlx_fn = Wrapper.mlx_sign,
            TIn = Union{AbstractFloat, Bool, Signed, Complex}, # TIn = Number \ Unsigned: sign broken on CPU for Unsigned on MLX <= 0.24.1, cf. https://github.com/ml-explore/mlx/issues/2023
            output_type = return_input_type,
            normalize = (a, TIn) -> a,
        ),
        :sin => (
            mlx_fn = Wrapper.mlx_sin,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> begin
                TIn <: Complex && return a
                return TIn.(round.(map(x -> iszero(x % π / 2) ? x + eps(Float32) : x, a)))
            end,
        ),
        :sinh => (
            mlx_fn = Wrapper.mlx_sinh,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        :sqrt => (
            mlx_fn = Wrapper.mlx_sqrt,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) ->
                TIn <: Real ? TIn.(ceil.(max.(eps(Float32), a))) : a,
        ),
        # mlx_square
        # mlx_stop_gradient
        :tan => (
            mlx_fn = Wrapper.mlx_tan,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        :tanh => (
            mlx_fn = Wrapper.mlx_tanh,
            TIn = Number,
            output_type = return_float_type,
            normalize = (a, TIn) -> a,
        ),
        # mlx_linalg_inv
        # :pinv => ( # TODO using LinearAlgebra
        #     mlx_fn = Wrapper.mlx_linalg_pinv,
        #     TIn = Number,
        #     output_type = return_input_type,
        #     normalize = (a, TIn) -> a,
        # ),
    )
end

end
