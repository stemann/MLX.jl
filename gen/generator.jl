using Clang.Generators

using Pkg.Artifacts
mlx_c_artifact_dir = artifact"MLX_C"

cd(@__DIR__)

include_dir = joinpath(mlx_c_artifact_dir, "include", "mlx")

options = load_options(joinpath(@__DIR__, "generator.toml"))

args = get_default_args()
push!(args, "-I$include_dir")
push!(args, "-I$(joinpath(mlx_c_artifact_dir, "include", "mlx"))")

headers = [
    joinpath(include_dir, header) for
    header in readdir(include_dir) if endswith(header, ".h")
]

ctx = create_context(headers, args, options)

build!(ctx)
