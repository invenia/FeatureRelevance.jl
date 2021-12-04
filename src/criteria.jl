"""
    Criterion

A callable type that determines the relevance between two features.
New criterion types must implement `evaluate(criterion, x, y)`.
"""
abstract type Criterion end

# NOTE: We may want to make the criterion types a kind of metric and extend Metrics.evaluate.

# These types are callable, so they can be used interchangeably with functions in a
# scoring method.
(criterion::Criterion)(args...) = evaluate(criterion, args...)

"""
    MutualInformation(; estimator="shrinkage")

Estimate the [mutual information]([normalized mutual information](https://en.wikipedia.org/wiki/Mutual_information)
between two sets of values using `InformationMeasures.get_mutual_information`.
"""
Base.@kwdef struct MutualInformation <: Criterion
    estimator::String = "shrinkage"
end

function evaluate(criterion::MutualInformation, x, y)
    return get_mutual_information(x, y; estimator=criterion.estimator)
end

"""
     NormalisedMutualInformation(; estimator="shrinkage")

Estimate the [normalized mutual information](https://en.wikipedia.org/wiki/Mutual_information#Normalized_variants)
between two sets of values using `InformationMeasures.get_mutual_information`.
"""
Base.@kwdef struct NormalisedMutualInformation <: Criterion
    estimator::String = "shrinkage"
end

function evaluate(criterion::NormalisedMutualInformation, x, y)
    kw = (; estimator=criterion.estimator)
    mi = get_mutual_information(x, y; kw...)
    return mi / sqrt(get_entropy(x; kw...) * get_entropy(y; kw...))
end

"""
     ConditionalMutualInformation(; estimator="shrinkage")

Estimate the [conditional mutual information](https://en.wikipedia.org/wiki/Conditional_mutual_information)
between two sets of values, conditioned on a third, using `InformationMeasures.get_conditional_mutual_information`.
"""
Base.@kwdef struct ConditionalMutualInformation <: Criterion
    estimator::String = "shrinkage"
end

function evaluate(criterion::ConditionalMutualInformation, x, y, z)
    return get_conditional_mutual_information(x, y, z; estimator=criterion.estimator)
end

# The types below for backward compatibility, but could just be functions.

"""
    PearsonCorrelation()

Use [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
by calling `cor(x, y)`.
"""
struct PearsonCorrelation <: Criterion end
evaluate(criterion::PearsonCorrelation, x, y) = cor(x, y)

"""
    SpearmanCorrelation()

Use [Spearman correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
by calling `corspearman(x, y)`.
"""
struct SpearmanCorrelation <: Criterion end
evaluate(criterion::SpearmanCorrelation, x, y) = corspearman(x, y)

"""
    RatioToShuffled(criterion::Criterion)

Compute the ratio of the `criterion(x, y)` and `criterion(x, shuffle(y))`, softened by the
stabiliser.
"""
struct RatioToShuffled <: Criterion
    criterion
end
function evaluate(criterion::RatioToShuffled, x, y, stabiliser=0.01)
    real = criterion.criterion(x, y) + stabiliser
    shuffled = criterion.criterion(x, shuffle(y)) + stabiliser
    return real / shuffled
end

"""
    RatioToLagged(criterion::Criterion; offset::Int=24)

Compute the ratio of `criterion(x[offset:end], y[offset:end])` and
`criterion(y[1:end-offset], y[offset:end])`.
"""
struct RatioToLagged <: Criterion
    criterion::Criterion
    offset::Int
end

RatioToLagged(criterion::Criterion; offset=24) = RatioToLagged(criterion, offset)

function evaluate(criterion::RatioToLagged, x, y)
    idx = criterion.offset:length(x)
    lag_idx = 1:length(x)-criterion.offset
    real = criterion.criterion(x[idx], y[idx])
    lagged = criterion.criterion(y[lag_idx], y[idx])
    return real / lagged
end
