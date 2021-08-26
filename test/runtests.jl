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
    selection,
    report

@testset "FeatureRelevance" begin
    include("criteria.jl")
    include("feature_selection.jl")
end
