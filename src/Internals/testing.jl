# Replaces leading elements of floating-point arrays by Inf, NaN etc., for testing
function with_nonfinite_values(a, ::Type{TIn}) where {TIn}
    if TIn <: AbstractFloat
        values = TIn[Inf, -Inf, NaN]
    elseif TIn <: Complex{<:AbstractFloat}
        values = TIn[Inf, complex(0, Inf), NaN, complex(0, NaN), -Inf]
    else
        return a
    end
    b = copy(a)
    for (i, value) in zip(eachindex(b), values)
        b[i] = value
    end
    return b
end

# Clamps real test inputs to [-1, 1], the domain of e.g. asin
function clamp_to_unit_interval(a, ::Type{TIn}) where {TIn}
    TIn <: Real || return a
    @static if VERSION < v"1.11" # clamp(x::Unsigned, -1, 1) throws an InexactError on Julia 1.10
        TIn <: Unsigned && return TIn.(clamp.(a, 0, 1))
    end
    return TIn.(clamp.(a, -1, 1))
end

# MLX 0.23.2 has no complex GPU kernels for some ops, e.g. asin
real_on_gpu(device_type) = device_type == DeviceTypeGPU ? Real : Number
