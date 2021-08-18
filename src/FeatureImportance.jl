module FeatureImportance
    using DataFrames
    using ProgressMeter
    using InformationMeasures
    using Statistics
    using StatsBase
    using Random
    using LightGBM
    include("feature_selection.jl")
    include("lightgbm.jl")
    include("predictive_power_score.jl")
    
    export GreedyMRMR, GreedyJMI
    export Top
    export MutualInformation, ConditionalMutualInformation, NormalisedMutualInformation
    export PredictivePowerScore
    export ShapleyValues, GainImportance, SplitImportance
    export PearsonCorrelation, SpearmanCorrelation
    export select
end # module
