using AxisKeys
using DataFrames
using FeatureRelevance
using InformationMeasures
using Plots
using Random
using Statistics
using Test

using FeatureRelevance:
    GreedyJMI,
    GreedyMRMR,
    MutualInformation,
    ConditionalMutualInformation,
    NormalisedMutualInformation,
    PearsonCorrelation,
    RatioToShuffled,
    RatioToLagged,
    SpearmanCorrelation,
    Top,
    evaluate,
    log_transform,
    relevance,
    report,
    selection

rng = MersenneTwister(1)

@testset "FeatureRelevance" begin
    include("algorithms.jl")
    include("criteria.jl")
    include("plotting.jl")
    include("preprocess.jl")
    include("relevance.jl")
    include("report.jl")

    # Only run LightGBM tests on linux by default
    Sys.islinux() && include("lightgbm.jl")
end
