using Documenter
using MLX

# The README doubles as the front page
cp(
    joinpath(@__DIR__, "..", "README.md"),
    joinpath(@__DIR__, "src", "index.md");
    force = true,
)

makedocs(;
    sitename = "MLX.jl",
    modules = [MLX],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["tutorials/quick-start.md"],
        "Reference" => ["reference/api.md"],
    ],
    checkdocs = :none,
)
