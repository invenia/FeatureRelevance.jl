module FeatureRelevance

using DataFrames
using InformationMeasures
using MLUtils: splitobs, kfolds
using Missings
using Random
using RecipesBase
using Requires
using Statistics
using StatsBase
using Tables

include("utils.jl")
include("criteria.jl")
include("algorithms.jl")
include("plotting.jl")
include("preprocess.jl")
include("relevance.jl")
include("report.jl")

# To avoid bogging down our package with dependencies we use requires to support
# optional features for now.
function __init__()
    @require LightGBM="7acf609c-83a4-11e9-1ffb-b912bcd3b04a" include("lightgbm.jl")
end

end # module
