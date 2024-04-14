using WGPUCompute
using Documenter

DocMeta.setdocmeta!(WGPUCompute, :DocTestSetup, :(using WGPUCompute); recursive=true)

makedocs(;
    modules=[WGPUCompute],
    authors="arhik <arhik23@gmail.com>",
    repo="https://github.com/JuliaWGPU/WGPUCompute.jl/blob/{commit}{path}#{line}",
    sitename="WGPUCompute.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaWGPU.github.io/WGPUCompute.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/JuliaWGPU/WGPUCompute.jl",
    devbranch="main",
)
