# An MLX binary op, and how to test it against Base
struct BinaryOp
    mlx_fn::Function
    input_type::Type
    compute_type::Union{Function, Nothing}
    test_input::Function
    test_type::Function
end

function BinaryOp(
    mlx_fn;
    input_type = Number,
    compute_type = nothing,
    test_input = (a, b) -> (a, b),
    test_type = device_type -> input_type,
)
    return BinaryOp(mlx_fn, input_type, compute_type, test_input, test_type)
end

function get_binary_ops()
    return Dict(
        :+ => BinaryOp(Wrapper.mlx_add),
        :- => BinaryOp(Wrapper.mlx_subtract),
        :* => BinaryOp(Wrapper.mlx_multiply),
        :/ => BinaryOp(
            Wrapper.mlx_divide;
            compute_type = (T, S) -> return_float_type(promote_type(T, S)),
            test_input = (a, b) -> (a, without_zeros(b)),
        ),
    )
end

# An operand of a broadcast of element type T, which MLX broadcasts to any shape
function broadcast_operand(::Type{T}, x::Number, ::Val) where {T}
    # Converts integers with %, as Julia's mixed integer arithmetic does, e.g. -1 to typemax(UInt64)
    y = x isa Base.BitInteger && T <: Base.BitInteger ? x % T : T(x)
    return MLXNumber(y)
end

# An operand of a broadcast with N dims, of element type T: Julia's broadcasting adds trailing
# singleton dims to operands with fewer dims, where MLX's adds leading ones
function broadcast_operand(::Type{T}, a::MLXArray{S, M}, ::Val{N}) where {T, S, M, N}
    a = S == T ? a : astype(T, a)
    M == N && return a
    return reshape(a, (size(a)..., ntuple(_ -> 1, Val(N - M))...))
end
