"""
    MLX.evaluate(arrays...)

Evaluates the given `MLXArray`s and `MLXNumber`s, which are otherwise only evaluated when
their values are needed, e.g. when indexing them.

Evaluating them together lets MLX compute the operations they share once.
"""
function evaluate(arrays::Union{MLXArray, MLXNumber}...)
    vector = Wrapper.mlx_vector_array_new()
    try
        for array in arrays
            Wrapper.mlx_vector_array_append_value(vector, array)
        end
        Wrapper.mlx_eval(vector)
    finally
        Wrapper.mlx_vector_array_free(vector)
    end
    return nothing
end
