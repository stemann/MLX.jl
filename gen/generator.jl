using Clang.Generators
using MLX_C_jll

cd(@__DIR__)

include_dir = joinpath(MLX_C_jll.artifact_dir, "include", "mlx", "c")

options = load_options(joinpath(@__DIR__, "generator.toml"))

args = get_default_args()
push!(args, "-I$(joinpath(MLX_C_jll.artifact_dir, "include"))")

# Ensure Float16 and BFloat16 methods are generated - even on platforms without support
push!(args, "-DHAS_FLOAT16")
push!(args, "-DHAS_BFLOAT16")

headers = [
    joinpath(include_dir, header) for
    header in readdir(include_dir) if endswith(header, ".h")
]

ctx = create_context(headers, args, options)

build!(ctx)
