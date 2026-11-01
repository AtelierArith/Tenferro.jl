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
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
)
