@enum ArrayStorageOrder begin
    ArrayStorageOrderColumn
    ArrayStorageOrderRow
end

size_to_strides_column(sz::Dims) = Base.size_to_strides(1, sz...)

size_to_strides_row(sz::Dims) = reverse(Base.size_to_strides(1, reverse(sz)...))

"""
    storage_order(a::AbstractArray; preferred_order::ArrayStorageOrder = ArrayStorageOrderColumn)

Returns the storage order of `a`, or `preferred_order` if `a` is a scalar or vector.
"""
function storage_order(
    a::AbstractArray{T, N}; preferred_order::ArrayStorageOrder = ArrayStorageOrderColumn
) where {T, N}
    if N < 2
        return preferred_order
    end

    if strides(a) == size_to_strides_column(size(a))
        return ArrayStorageOrderColumn
    elseif strides(a) == size_to_strides_row(size(a))
        return ArrayStorageOrderRow
    else
        throw(
            ArgumentError("Unexpected strides $(strides(a)) for array of size $(size(a))")
        )
    end
end
