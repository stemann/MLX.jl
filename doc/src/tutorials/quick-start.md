# Quick start

Make an `MLXArray` from a Julia array:

```jldoctest quick-start
julia> using MLX

julia> a = MLXArray([1, 2, 3, 4]);

julia> size(a)
(4,)

julia> eltype(a)
Int64

julia> b = MLXArray(Float32[1, 2, 3, 4]);

julia> eltype(b)
Float32
```

Julia's floating-point literals are `Float64`, which MLX does not support on GPUs, hence `Float32[1, 2, 3, 4]` rather than `[1.0, 2.0, 3.0, 4.0]`.

Operations in MLX are lazy: their results are not computed until they are needed. To force an array to be evaluated, use [`MLX.evaluate`](@ref). Arrays are also evaluated when their values are needed, e.g. when displaying them, or converting them to `Array`s:

```jldoctest quick-start
julia> c = a + b; # c is not yet evaluated

julia> MLX.evaluate(c) # Evaluates c

julia> c = a + b;

julia> c # Displaying c also evaluates it
4-element MLXVector{Float32}:
 2.0
 4.0
 6.0
 8.0

julia> c = a + b;

julia> Array(c) # So does converting it
4-element Vector{Float32}:
 2.0
 4.0
 6.0
 8.0
```

See MLX's page on [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) for more details.
