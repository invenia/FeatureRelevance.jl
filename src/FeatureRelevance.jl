module FeatureRelevance

using Distributed
using InformationMeasures
using Missings
using Random
using Statistics
using StatsBase
using Tables

include("criteria.jl")
include("relevance.jl")
include("algorithms.jl")
include("report.jl")

end # module
