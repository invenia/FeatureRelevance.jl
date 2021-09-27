# Contains all package features which depend on LightGBM.
# Since that package requires manually installing the lightgbm binary dependency and
# several MLJ packages we're keeping this as an optional dependency for now.
# https://gitlab.invenia.ca/invenia/research/FeatureRelevance.jl/-/issues/6
using LightGBM

"""
    PredictivePowerScore(; k=4)

Estimate the [predictive power score](https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598)
between two sets of values using a `LightGBM.LGBMRegression` model and `median` naive estimators.
Note that categorical variables are not currently supported.

- `k` is the number of folds used in the decision tree training.
"""
Base.@kwdef struct PredictivePowerScore <: Criterion
    k::Int = 4
end

# Dispatch to convert `x` vector to a matrix for LightGBM.
function evaluate(criterion::PredictivePowerScore, x::Vector{<:Real}, y)
    return evaluate(criterion, reshape(x, (length(x), 1)), y)
end

function evaluate(criterion::PredictivePowerScore, x::Matrix{<:Real}, y::Vector{<:Real})
    n = 0
    model_ae = 0.0
    naive_ae = 0.0

    folds = kfolds((x, y); k=criterion.k, obsdim=:first)

    for ((train_X, train_y), (test_X, test_y)) in folds
        estimator = LGBMRegression()

        # LightGBM doesn't like array views, so we need to call `collect`
        LightGBM.fit!(estimator, collect(train_X), collect(train_y))
        preds = LightGBM.predict(estimator, collect(test_X))

        model_ae += sum(abs.(preds .- test_y))
        naive_ae += sum(abs.(median(train_y) .- test_y))
        n += length(test_y)
    end

    model_mae = model_ae / n
    naive_mae = naive_ae / n

    return model_mae > naive_mae ? 0.0 : 1 - (model_mae / naive_mae)
end

"""
    RandomForest(; importance_type, iterations=0)

Fits a `LighGBM.LGBMRegression` estimator to a given set of features / targets, and scores
the features using [`LGBM_BoosterFeatureImportance`](https://lightgbm.readthedocs.io/en/latest/C-API.html?highlight=gain#c.LGBM_BoosterFeatureImportance).

# Arguments

- `importance_type::Symbol`: Whether to use `:gain` or `:split` importance scoring.
- `iterations=0`: Maximum number of iterations/boosting to consider. The default
value of 0 means that a single decision tree is used.
"""
Base.@kwdef struct RandomForest <: Algorithm
    importance_type::Symbol
    iterations::UInt = 0

    function RandomForest(importance_type::Symbol, iterations=0)
        # This seemed simpler than an enum type or multiple wrappers
        if importance_type ∉ (:split, :gain)
            throw(ArgumentError("Supported importance types are :split or :gain"))
        end

        return new(importance_type, UInt(iterations))
    end
end

GainImportance(; iterations=0) = RandomForest(:gain, iterations)
SplitImportance(; iterations=0) = RandomForest(:split, iterations)

function selection(alg::RandomForest, target, features)
    # NOTE: If we have more than 1 target column LightGBM will error with a
    # size mismatch error.
    return selection(alg, vec(_to_real_array(target)), _to_real_array(features))
end

function selection(alg::RandomForest, target::Vector{<:Real}, features::Matrix{<:Real})
    estimator = LGBMRegression()
    LightGBM.fit!(estimator, features, target)

    # Rather than having a separate wrapper for each importance type we just pass
    # a 0/1 for split/gain
    # https://lightgbm.readthedocs.io/en/latest/C-API.html?highlight=gain#c.C_API_FEATURE_IMPORTANCE_GAIN
    scores = LightGBM.feature_importance_wrapper(
        estimator,
        alg.importance_type === :gain,
        alg.iterations,
    )

    return collect(eachindex(scores)), scores
end

# Utility function for coercing tables and arrays into the correct LightGBM type
_to_real_array(data::Array{<:Real}) = data
function _to_real_array(data)
    X = Tables.istable(data) ? Tables.matrix(data) : data
    return Matrix{Float64}(coalesce(X, NaN))
end
