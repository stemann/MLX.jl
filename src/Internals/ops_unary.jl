# An MLX unary op, and how to test it against Base
struct UnaryOp
    mlx_fn::Function
    input_type::Type
    output_type::Function
    test_input::Function
    test_type::Function
end

function UnaryOp(
    mlx_fn;
    input_type = Number,
    output_type = return_input_type,
    test_input = (a, T) -> a,
    test_type = device_type -> input_type,
)
    return UnaryOp(mlx_fn, input_type, output_type, test_input, test_type)
end

function get_unary_ops()
    RealExceptBool = Union{AbstractFloat, Signed, Unsigned}
    return Dict(
        :abs => UnaryOp(Wrapper.mlx_abs; output_type = return_real_type),
        :acos => UnaryOp(
            Wrapper.mlx_arccos;
            output_type = return_float_type,
            test_input = clamp_to_unit_interval,
            test_type = real_on_gpu,
        ),
        :acosh => UnaryOp(
            Wrapper.mlx_arccosh;
            output_type = return_float_type,
            test_input = (a, T) -> begin
                T <: Complex && return a
                T == Bool && return trues(size(a)) # acosh(false) is undefined
                return T.(abs.(a) .+ 1)
            end,
            test_type = real_on_gpu,
        ),
        :asin => UnaryOp(
            Wrapper.mlx_arcsin;
            output_type = return_float_type,
            test_input = clamp_to_unit_interval,
            test_type = real_on_gpu,
        ),
        :asinh => UnaryOp(
            Wrapper.mlx_arcsinh;
            output_type = return_float_type,
            test_type = real_on_gpu,
        ),
        :atan => UnaryOp(
            Wrapper.mlx_arctan; output_type = return_float_type, test_type = real_on_gpu
        ),
        :atanh => UnaryOp(
            Wrapper.mlx_arctanh;
            output_type = return_float_type,
            test_input = clamp_to_unit_interval,
            test_type = real_on_gpu,
        ),
        :~ => UnaryOp(Wrapper.mlx_bitwise_invert; input_type = Integer),
        :ceil => UnaryOp(
            Wrapper.mlx_ceil;
            input_type = Real, # MLX 0.23.2: [floor] Not supported for complex64
        ),
        :conj => UnaryOp(Wrapper.mlx_conjugate),
        :cos => UnaryOp(Wrapper.mlx_cos; output_type = return_float_type),
        :cosh => UnaryOp(Wrapper.mlx_cosh; output_type = return_float_type),
        :rad2deg => UnaryOp(Wrapper.mlx_degrees; output_type = return_float_type),
        :exp => UnaryOp(Wrapper.mlx_exp; output_type = return_float_type),
        :expm1 => UnaryOp(
            Wrapper.mlx_expm1;
            input_type = Real, # MLX 0.23.2: expm1 of complex inputs ignores the imaginary part; later versions reject them
            output_type = return_float_type,
        ),
        :floor => UnaryOp(
            Wrapper.mlx_floor;
            input_type = Real, # MLX 0.23.2: [floor] Not supported for complex64
        ),
        :imag => UnaryOp(Wrapper.mlx_imag; output_type = return_real_type),
        :isfinite => UnaryOp(
            Wrapper.mlx_isfinite;
            output_type = (::Type) -> Bool,
            test_input = with_nonfinite_values,
            test_type = device_type -> Real, # MLX 0.23.2: isfinite(complex(0, Inf)) is true
        ),
        :isinf => UnaryOp(
            Wrapper.mlx_isinf;
            output_type = (::Type) -> Bool,
            test_input = with_nonfinite_values,
            test_type = device_type -> Real, # MLX 0.23.2: isinf(complex(0, Inf)) is false
        ),
        :isnan => UnaryOp(
            Wrapper.mlx_isnan;
            output_type = (::Type) -> Bool,
            test_input = with_nonfinite_values,
        ),
        :log => UnaryOp(
            Wrapper.mlx_log;
            output_type = return_float_type,
            test_input = (a, T) -> T <: Real ? T.(ceil.(max.(eps(Float32), a))) : a,
            test_type = real_on_gpu,
        ),
        :log10 => UnaryOp(
            Wrapper.mlx_log10;
            output_type = return_float_type,
            test_input = (a, T) -> T <: Real ? T.(ceil.(max.(eps(Float32), a))) : a,
            test_type = real_on_gpu,
        ),
        :log1p => UnaryOp(
            Wrapper.mlx_log1p;
            output_type = return_float_type,
            test_input = (a, T) -> T <: Real ? T.(ceil.(max.(eps(Float32), a))) : a,
            test_type = device_type -> Real, # MLX 0.23.2: log1p of complex inputs ignores the imaginary part
        ),
        :log2 => UnaryOp(
            Wrapper.mlx_log2;
            output_type = return_float_type,
            test_input = (a, T) -> T <: Real ? T.(ceil.(max.(eps(Float32), a))) : a,
            test_type = device_type -> Real, # MLX 0.23.2: log2 of complex inputs ignores the imaginary part
        ),
        :! => UnaryOp(Wrapper.mlx_logical_not; input_type = Bool),
        :- => UnaryOp(
            Wrapper.mlx_negative;
            input_type = Union{RealExceptBool, Complex{<:AbstractFloat}}, # MLX 0.23.2: [negative] Not supported for bool, use logical_not instead.
        ),
        :one => UnaryOp(Wrapper.mlx_ones_like),
        :deg2rad => UnaryOp(Wrapper.mlx_radians; output_type = return_float_type),
        :real => UnaryOp(Wrapper.mlx_real; output_type = return_real_type),
        :inv => UnaryOp(Wrapper.mlx_reciprocal; output_type = return_float_type),
        :round => UnaryOp(
            # round.(a; digits) falls back to Base's generic broadcasting
            (res, a, s) -> Wrapper.mlx_round(res, a, 0, s);
            # MLX 0.23.2 has no GPU kernels for rounding integers
            test_type = device_type ->
                device_type == DeviceTypeGPU ? Union{AbstractFloat, Complex} : Number,
        ),
        :sign => UnaryOp(
            Wrapper.mlx_sign;
            input_type = Union{AbstractFloat, Bool, Signed, Complex}, # T = Number \ Unsigned: sign broken on CPU for Unsigned on MLX <= 0.24.1, cf. https://github.com/ml-explore/mlx/issues/2023
        ),
        :sin => UnaryOp(Wrapper.mlx_sin; output_type = return_float_type),
        :sinh => UnaryOp(Wrapper.mlx_sinh; output_type = return_float_type),
        :sqrt => UnaryOp(
            Wrapper.mlx_sqrt;
            output_type = return_float_type,
            test_input = (a, T) -> T <: Real ? T.(ceil.(max.(eps(Float32), a))) : a,
            test_type = real_on_gpu,
        ),
        :abs2 => UnaryOp(Wrapper.mlx_square; input_type = Real), # MLX 0.23.2: square(z) is z^2, not abs2(z), for complex z
        :tan => UnaryOp(Wrapper.mlx_tan; output_type = return_float_type),
        :tanh => UnaryOp(Wrapper.mlx_tanh; output_type = return_float_type),
        :zero => UnaryOp(Wrapper.mlx_zeros_like),
    )
end
