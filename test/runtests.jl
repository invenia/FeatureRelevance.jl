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
    Top,
    relevance,
    report,
    selection

rng = MersenneTwister(1)

@testset "FeatureRelevance" begin
    include("algorithms.jl")
    include("criteria.jl")
    include("report.jl")
end
