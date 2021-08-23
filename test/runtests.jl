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
    _select_single_target,
    select

include("feature_selection.jl")
