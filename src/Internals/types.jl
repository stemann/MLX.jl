function return_input_type(::Type{TIn}) where {TIn}
    return TIn
end

function return_float_type(::Type{TIn}) where {TIn}
    return TIn <: Integer ? Float32 : TIn
end

function return_real_type(::Type{TIn}) where {TIn}
    return TIn <: Complex{<:AbstractFloat} ? real(TIn) : TIn
end
