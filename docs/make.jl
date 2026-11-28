using Documenter
using DocumenterVitepress
using Tenferro

makedocs(;
    modules = [Tenferro],
    sitename = "Tenferro.jl",
    authors = "Tenferro.jl contributors",
    remotes = nothing,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "https://github.com/AtelierArith/Tenferro.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Automatic differentiation" => "autodiff.md",
        "Operation contracts" => "operations.md",
        "Performance" => "performance.md",
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/AtelierArith/Tenferro.jl.git",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    devurl = "dev",
    push_preview = false,
)
