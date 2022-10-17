using WGPUCompute
using Documenter

DocMeta.setdocmeta!(WGPUCompute, :DocTestSetup, :(using WGPUCompute); recursive=true)

makedocs(;
    modules=[WGPUCompute],
    authors="arhik <arhik23@gmail.com>",
    repo="https://github.com/arhik/WGPUCompute.jl/blob/{commit}{path}#{line}",
    sitename="WGPUCompute.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://arhik.github.io/WGPUCompute.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/arhik/WGPUCompute.jl",
    devbranch="main",
)
