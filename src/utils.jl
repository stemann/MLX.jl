@enum StorageOrder begin
    StorageOrderColumn
    StorageOrderRow
end

size_to_strides_column(sz::Dims) = Base.size_to_strides(1, sz...)

size_to_strides_row(sz::Dims) = reverse(Base.size_to_strides(1, reverse(sz)...))

"""
    storage_order(a::AbstractArray, preferred_order::StorageOrder = StorageOrderColumn)

Returns the storage order of `a`, or `preferred_order` if `a` is a scalar or vector.
"""
function storage_order(
    a::AbstractArray{T, N}, preferred_order::StorageOrder = StorageOrderColumn
) where {T, N}
    if N < 2
        return preferred_order
    end

    if strides(a) == size_to_strides_column(size(a))
        return StorageOrderColumn
    elseif strides(a) == size_to_strides_row(size(a))
        return StorageOrderRow
    else
        throw(
            ArgumentError("Unexpected strides $(strides(a)) for array of size $(size(a))")
        )
    end
end
