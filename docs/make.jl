using LightGBM
using FeatureRelevance
using Documenter

DocMeta.setdocmeta!(
    FeatureRelevance, :DocTestSetup, :(using FeatureRelevance); recursive=true
)

makedocs(;
    modules=[FeatureRelevance],
    authors="Invenia Technical Computing Corporation",
    repo="https://gitlab.invenia.ca/invenia/research/FeatureRelevance.jl/blob/{commit}{path}#{line}",
    sitename="FeatureRelevance.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true", assets=String[]),
    pages=[
        "Home" => "index.md",
        "Usage" => "usage.md",
        "Recommendations" => "recommendations.md",
        "FAQ" => "faq.md",
        "API" => [
            "Reports" => "api/report.md",
            "Criteria" => "api/criteria.md",
            "Algorithms" => "api/algorithms.md",
            "Extensions" => "api/extensions.md",
        ],
    ],
    checkdocs=:exports,
    strict=true,
)
