# Julia and MLX semantics

Base functions on `MLXArray`s follow Julia's semantics, so that `MLXArray`s can be used like `Array`s. This page lists where they differ from those on `Array`, and from MLX's semantics, which matters when porting code written for MLX.

```jldoctest semantics; output = false
using MLX

# output

```

## Differences from `Array`

### Floating-point results

Functions with floating-point results give `Float32` for integer inputs, where `Array` gives `Float64`, as MLX does not support `Float64` on GPUs:

```jldoctest semantics
julia> MLXArray([1, 2]) / 2
2-element MLXVector{Float32}:
 0.5
 1.0
```

### Complex numbers

MLX 0.23.2 does not support `ComplexF64`, so operations for which `Array` gives `ComplexF64` throw:

```jldoctest semantics
julia> MLXArray(ComplexF32[1, 2]) .* 2.0
ERROR: ArgumentError: Unsupported type: ComplexF64
```

### Lazy evaluation

Operations are computed when their results are needed, or by [`MLX.evaluate`](@ref), see the [quick start](../tutorials/quick-start.md).

### Zero-dimensional results

Broadcasts with 0-dimensional results give `MLXNumber`s, which are also evaluated lazily, where `Array` gives numbers:

```jldoctest semantics
julia> MLXArray(fill(2f0)) .* 3
MLXNumber(6.0f0)
```

### Memory layout

Elements are stored in row-major order, so strides differ from those of `Array`, whereas linear indexing is in column-major order, as for `Array`:

```jldoctest semantics
julia> a = MLXArray([1 2; 3 4]);

julia> a[2]
3

julia> strides(a)
(2, 1)
```

## Differences from MLX

### Broadcasting

Julia aligns the dimensions of arrays from the left, where MLX, like NumPy, aligns them from the right:

```jldoctest semantics
julia> size(MLXArray(ones(Float32, 3, 2)) .+ MLXArray(ones(Float32, 3)))
(3, 2)

julia> MLXArray(ones(Float32, 3, 2)) .+ MLXArray(ones(Float32, 2))
ERROR: DimensionMismatch: arrays could not be broadcast to a common size: a has axes Base.OneTo(3) and b has axes Base.OneTo(2)
```

### Type promotion

Julia's type promotion applies, e.g. `Bool` and `Bool` give `Int64`, where MLX gives `bool`, and `UInt64` and `Int64` give `UInt64`, where MLX gives `float32`:

```jldoctest semantics
julia> eltype(MLXArray([true]) .+ MLXArray([true]))
Int64

julia> eltype(MLXArray(UInt64[1]) .+ MLXArray(Int64[1]))
UInt64
```

### Integers

Integer literals are `Int64`, on 64-bit platforms, where MLX's are `int32`:

```jldoctest semantics
julia> eltype(MLXArray([1, 2]))
Int64
```

### Indexing

Indices start at 1, where MLX's start at 0:

```jldoctest semantics
julia> MLXArray([1 2; 3 4])[1, 2]
2
```
