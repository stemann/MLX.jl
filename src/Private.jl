module Private

using ..Wrapper

const unary_ops = Dict(
    :abs => Wrapper.mlx_abs,
    :acos => Wrapper.mlx_arccos,
    :acosh => Wrapper.mlx_arccosh,
    :asin => Wrapper.mlx_arcsin,
    :asinh => Wrapper.mlx_arcsinh,
    :atan => Wrapper.mlx_arctan,
    :atanh => Wrapper.mlx_arctanh,
    :sortperm => Wrapper.mlx_argsort_all, # TODO check if this is correct
    # mlx_atleast_1d
    # mlx_atleast_2d
    # mlx_atleast_3d
    :ceil => Wrapper.mlx_ceil,
    :conj => Wrapper.mlx_conjugate,
    :copy => Wrapper.mlx_copy,
    :cos => Wrapper.mlx_cos,
    :cosh => Wrapper.mlx_cosh,
    :rad2deg => Wrapper.mlx_degrees,
    # mlx_erf
    # mlx_erfinv
    :exp => Wrapper.mlx_exp,
    :expm1 => Wrapper.mlx_expm1,
    :floor => Wrapper.mlx_floor,
    :imag => Wrapper.mlx_imag,
    :isfinite => Wrapper.mlx_isfinite,
    :isinf => Wrapper.mlx_isinf,
    :isnan => Wrapper.mlx_isnan,
    # mlx_isneginf
    # mlx_isposinf
    :log => Wrapper.mlx_log,
    :log10 => Wrapper.mlx_log10,
    :log1p => Wrapper.mlx_log1p,
    :log2 => Wrapper.mlx_log2,
    :! => Wrapper.mlx_logical_not,
    :- => Wrapper.mlx_negative,
    :deg2rad => Wrapper.mlx_radians,
    :real => Wrapper.mlx_real,
    :inv => Wrapper.mlx_reciprocal, # TODO check if this is correct, notably wrt. mlx_linalg_inv
    # mlx_rsqrt
    # mlx_sigmoid
    :sign => Wrapper.mlx_sign,
    :sin => Wrapper.mlx_sin,
    :sinh => Wrapper.mlx_sinh,
    :sort => Wrapper.mlx_sort_all, # TODO check if this is correct
    :sqrt => Wrapper.mlx_sqrt,
    # mlx_square
    # :dropdims => Wrapper.mlx_squeeze,
    # mlx_stop_gradient
    :tan => Wrapper.mlx_tan,
    :tanh => Wrapper.mlx_tanh,
    :transpose => Wrapper.mlx_transpose_all,
    # mlx_zeros_like
    # mlx_random_split
    # mlx_linalg_inv
    # :pinv => Wrapper.mlx_linalg_pinv, # TODO using LinearAlgebra
    # :qr => Wrapper.mlx_linalg_qr, # TODO using LinearAlgebra
)

end
