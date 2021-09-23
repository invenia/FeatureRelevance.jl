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
