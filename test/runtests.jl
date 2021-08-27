using DataFrames
using FeatureRelevance
using InformationMeasures
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
    SpearmanCorrelation,
    Top,
    evaluate,
    relevance,
    report,
    selection

rng = MersenneTwister(1)

@testset "FeatureRelevance" begin
    include("algorithms.jl")
    include("criteria.jl")
    include("relevance.jl")
    include("report.jl")
end
