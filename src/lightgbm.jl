# Contains all package features which depend on LightGBM.
# Since that package requires manually installing the lightgbm binary dependency and
# several MLJ packages we're keeping this as an optional dependency for now.
# https://gitlab.invenia.ca/invenia/research/FeatureRelevance.jl/-/issues/6
using LightGBM

"""
    PredictivePowerScore(; k=4, verbosity=-1)

Estimate the [predictive power score](https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598)
between two sets of values using a `LightGBM.LGBMRegression` model and `median` naive estimators.
Note that categorical variables are not currently supported.

- `k` is the number of folds used in the decision tree training.
- `verbosity` the verbosity level to pass to LightGBM
"""
Base.@kwdef struct PredictivePowerScore <: Criterion
    k::Int = 4
    verbosity::Int = -1
end

# Dispatch to convert `x` vector to a matrix for LightGBM.
function evaluate(criterion::PredictivePowerScore, x::Vector{<:Real}, y)
    return evaluate(criterion, reshape(x, (length(x), 1)), y)
end

function evaluate(criterion::PredictivePowerScore, x::Matrix{<:Real}, y::Vector{<:Real})
    n = 0
    model_ae = 0.0
    naive_ae = 0.0

    verbosity = criterion.verbosity
    folds = kfolds((x', y'); k=criterion.k)

    for (train, test) in folds
        train_X, train_y = permutedims.(train)
        test_X, test_y = permutedims.(test)

        estimator = LGBMRegression()

        LightGBM.fit!(estimator, train_X, vec(train_y); verbosity)
        preds = LightGBM.predict(estimator, test_X; verbosity)

        model_ae += sum(abs.(preds .- test_y))
        naive_ae += sum(abs.(median(train_y) .- test_y))
        n += length(test_y)
    end

    model_mae = model_ae / n
    naive_mae = naive_ae / n

    return model_mae > naive_mae ? 0.0 : 1 - (model_mae / naive_mae)
end

"""
    GradientBoostedImportance(; importance_type, iterations=0, positive=false, verbosity=-1)

Fits a `LighGBM.LGBMRegression` estimator to a given set of features / targets, and scores
the features using [`LGBM_BoosterFeatureImportance`](https://lightgbm.readthedocs.io/en/latest/C-API.html?highlight=gain#c.LGBM_BoosterFeatureImportance).

# Arguments

- `importance_type::Symbol`: Whether to use `:gain` or `:split` importance scoring.
- `iterations=0`: Maximum number of iterations/boosting to consider. The default
value of 0 means that a single decision tree is used.
- `positive`: Only return positive (non-redundant) scores
"""
Base.@kwdef struct GradientBoostedImportance <: Algorithm
    importance_type::Symbol
    iterations::UInt=0
    positive::Bool=false
    verbosity::Int=-1

    function GradientBoostedImportance(importance_type::Symbol, iterations=0, positive=false, verbosity=-1)
        # This seemed simpler than an enum type or multiple wrappers
        if importance_type ∉ (:split, :gain)
            throw(ArgumentError("Supported importance types are :split or :gain"))
        end

        return new(importance_type, UInt(iterations), positive, verbosity)
    end
end

GainImportance(; kwargs...) = GradientBoostedImportance(; importance_type=:gain, kwargs...)
SplitImportance(; kwargs...) = GradientBoostedImportance(; importance_type=:split, kwargs...)

function selection(
    alg::GradientBoostedImportance,
    features::Matrix{<:Real},
    target::Vector{<:Real},
)
    verbosity=alg.verbosity
    estimator = LGBMRegression()
    LightGBM.fit!(estimator, features, target; verbosity)

    # Rather than having a separate wrapper for each importance type we just pass
    # a 0/1 for split/gain
    # https://lightgbm.readthedocs.io/en/latest/C-API.html?highlight=gain#c.C_API_FEATURE_IMPORTANCE_GAIN
    selected_scores = LightGBM.feature_importance_wrapper(
        estimator,
        alg.importance_type === :gain,
        alg.iterations,
    )
    selected_indices = collect(eachindex(selected_scores))

    if alg.positive
        pos_idx = findall(>(0.0), selected_scores)
        if length(pos_idx) < length(selected_scores)
            return selected_indices[pos_idx], selected_scores[pos_idx]
        end
    end

    return selected_indices, selected_scores
end


"""
    ShapleyValues(; train_size=0.8, positive=false, verbosity=-1)

Fits a `LighGBM.LGBMRegression` estimator on a training partition of the provided features / targets.
SHAP scores are calculated by averaging the predicted SHAP contributions
([`predict_type=3`](https://github.com/IQVIA-ML/LightGBM.jl/blob/master/src/predict.jl#L12)) on the remaining test features.

Reminder: Only the magnitude (not the direction) of a SHAP value determines the relevance of a feature.

# Arguments

- `train_size::Float64=0.2`: Train partition ratio size. The default value of 0.8 represents a 80/20 train/test split.
- `positive::Bool=false`: Only return non-zero scores. SHAP scores can be positive or negative.
- `verbosity::Int=-1`: The verbosity level to pass to LightGBM
"""
Base.@kwdef struct ShapleyValues <: Algorithm
    train_size::Float64=0.8
    positive::Bool=false
    verbosity::Int=-1
end

function selection(alg::ShapleyValues, features::Matrix{<:Real}, target::Vector{<:Real})
    # Split test/train data
    (train, test) = splitobs((
        features', target');
        at=alg.train_size,
    )
    train_X, train_y = permutedims.(train)
    test_X, _ =  permutedims.(test)

    # Train and predict with LGBMRegression and predict_type=3
    verbosity=alg.verbosity
    estimator = LGBMRegression()
    LightGBM.fit!(estimator, train_X, vec(train_y); alg.verbosity)
    preds = LightGBM.predict(estimator, test_X; predict_type=3)  # 3 indicates SHAP

    # Reshape the output vector back to a matrix, so we can take the median for each feature.
    # The extra value per observation represents a bias value.
    # https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForCSC
    nobs, nfeatures = size(test_X)
    selected_scores = mean(reshape(preds, (nfeatures+1, nobs)); dims=2)[1:nfeatures]
    selected_indices = collect(eachindex(selected_scores))

    # Since SHAP values can be positive or negative the `positive` kwarg will remove values
    # that are exactly 0.0.
    if alg.positive
        idx = findall(!iszero, selected_scores)
        if length(idx) < length(selected_scores)
            return selected_indices[idx], selected_scores[idx]
        end
    end

    return selected_indices, selected_scores
end

# Utility function for coercing tables and arrays into the correct LightGBM type
function selection(alg::Union{GradientBoostedImportance,ShapleyValues}, features, target)
    # NOTE: If we have more than 1 target column LightGBM will error with a
    # size mismatch error.
    return selection(alg, _to_real_array(features), vec(_to_real_array(target)))
end

_to_real_array(data::Array{<:Real}) = data
_to_real_array(data::Base.Generator) = _to_real_array(reduce(hcat, data))
_to_real_array(data::AbstractVector{<:AbstractVector}) = _to_real_array(reduce(hcat, data))
_to_real_array(data::AbstractVector{<:Real}) = _to_real_array(reshape(data, length(data), 1))
function _to_real_array(data)
    X = Tables.istable(data) ? Tables.matrix(data) : data
    return Matrix{Float64}(coalesce(X, NaN))
end
