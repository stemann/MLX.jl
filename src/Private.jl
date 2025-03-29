module Private

using ..Wrapper

function return_input_type(::Type{TIn}) where {TIn}
    return TIn
end
function get_unary_ops()
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
            output_type = (::Type) -> Float32, # TODO: Float64 unsupported by MLX C 0.1.1
            preserves_type = false,
            normalize = (a, TIn) -> floor.(TIn, a ./ maximum(a)),
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
            normalize = (a, TIn) -> floor.(TIn, a ./ maximum(a)),
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
            output_type = (::Type) -> Float32, # TODO: Float64 unsupported by MLX C 0.1.1
            preserves_type = false,
            normalize = (a, TIn) -> a,
        ),
        :atanh => (
            mlx_fn = Wrapper.mlx_arctanh,
            TIn = AbstractFloat, # in testing, atanh differs from mlx_arctanh wrt. Integer, normalize fails for Complex{<:AbstractFloat}
            output_type = (::Type) -> Float32, # TODO: Float64 unsupported by MLX C 0.1.1
            preserves_type = false,
            normalize = (a, TIn) -> floor.(TIn, a ./ maximum(a)),
        ),
        :sortperm => (
            mlx_fn = Wrapper.mlx_argsort_all, # TODO check if this is correct
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        # mlx_atleast_1d
        # mlx_atleast_2d
        # mlx_atleast_3d
        :ceil => (
            mlx_fn = Wrapper.mlx_ceil,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :conj => (
            mlx_fn = Wrapper.mlx_conjugate,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :copy => (
            mlx_fn = Wrapper.mlx_copy,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :cos => (
            mlx_fn = Wrapper.mlx_cos,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :cosh => (
            mlx_fn = Wrapper.mlx_cosh,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :rad2deg => (
            mlx_fn = Wrapper.mlx_degrees,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        # mlx_erf
        # mlx_erfinv
        :exp => (
            mlx_fn = Wrapper.mlx_exp,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :expm1 => (
            mlx_fn = Wrapper.mlx_expm1,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :floor => (
            mlx_fn = Wrapper.mlx_floor,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :imag => (
            mlx_fn = Wrapper.mlx_imag,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :isfinite => (
            mlx_fn = Wrapper.mlx_isfinite,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :isinf => (
            mlx_fn = Wrapper.mlx_isinf,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :isnan => (
            mlx_fn = Wrapper.mlx_isnan,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        # mlx_isneginf
        # mlx_isposinf
        :log => (
            mlx_fn = Wrapper.mlx_log,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :log10 => (
            mlx_fn = Wrapper.mlx_log10,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :log1p => (
            mlx_fn = Wrapper.mlx_log1p,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :log2 => (
            mlx_fn = Wrapper.mlx_log2,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :! => (
            mlx_fn = Wrapper.mlx_logical_not,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :- => (
            mlx_fn = Wrapper.mlx_negative,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :deg2rad => (
            mlx_fn = Wrapper.mlx_radians,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :real => (
            mlx_fn = Wrapper.mlx_real,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :inv => (
            mlx_fn = Wrapper.mlx_reciprocal, # TODO check if this is correct, notably wrt. mlx_linalg_inv
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        # mlx_rsqrt
        # mlx_sigmoid
        :sign => (
            mlx_fn = Wrapper.mlx_sign,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :sin => (
            mlx_fn = Wrapper.mlx_sin,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :sinh => (
            mlx_fn = Wrapper.mlx_sinh,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :sort => (
            mlx_fn = Wrapper.mlx_sort_all, # TODO check if this is correct
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :sqrt => (
            mlx_fn = Wrapper.mlx_sqrt,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        # mlx_square
        # :dropdims => (
        #     mlx_fn = Wrapper.mlx_squeeze,
        #     TIn = Real,
        #     output_type = return_input_type,
        #     preserves_type = true,
        #     normalize = (a, TIn) -> a,
        # ),
        # mlx_stop_gradient
        :tan => (
            mlx_fn = Wrapper.mlx_tan,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :tanh => (
            mlx_fn = Wrapper.mlx_tanh,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        :transpose => (
            mlx_fn = Wrapper.mlx_transpose_all,
            TIn = Real,
            output_type = return_input_type,
            preserves_type = true,
            normalize = (a, TIn) -> a,
        ),
        # mlx_zeros_like
        # mlx_random_split
        # mlx_linalg_inv
        # :pinv => ( # TODO using LinearAlgebra
        #     mlx_fn = Wrapper.mlx_linalg_pinv,
        #     TIn = Real,
        #     output_type = return_input_type,
        #     preserves_type = true,
        #     normalize = (a, TIn) -> a,
        # ),
        # :qr => ( # TODO using LinearAlgebra
        #     mlx_fn = Wrapper.mlx_linalg_qr,
        #     TIn = Real,
        #     output_type = return_input_type,
        #     preserves_type = true,
        #     normalize = (a, TIn) -> a,
        # ),
    )
end

end
