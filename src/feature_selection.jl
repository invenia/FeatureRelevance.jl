"""
What determines the relevance between features.
"""
abstract type FeatureRelevanceCriterion end

struct MutualInformation <: FeatureRelevanceCriterion end
struct ConditionalMutualInformation <: FeatureRelevanceCriterion end
struct NormalisedMutualInformation <: FeatureRelevanceCriterion end
struct PearsonCorrelation <: FeatureRelevanceCriterion end
struct SpearmanCorrelation <: FeatureRelevanceCriterion end

"""
How do we select features.
"""
abstract type FeatureSelectionMethod end
abstract type GreedyMethod <: FeatureSelectionMethod end
abstract type RandomForestMethod <: FeatureSelectionMethod end

"""
    Top(N, ::FeatureRelevanceCriterion)

Select the top N relevant features.
This is fast but might result in redundant features.
Mutual information maximisation (MIM).
"""
struct Top <: FeatureSelectionMethod
    N::Int
    criterion::FeatureRelevanceCriterion
end

"""
    GreedyMRMR(N)

Greedy selection taking into account pairwise dependence of features, but assuming pairwise
class-condtional independence.
Maximum relevancy minimum redundancy (MRMR).
"""
struct GreedyMRMR <: GreedyMethod
    N::Int
end

"""
    GreedyJMI(N)

Greedy selection taking into account pairwise dependence of features, and also taking into
account pairwise class-condtional dependence.
Joint mutual information (JMI).
"""
struct GreedyJMI <: GreedyMethod
    N::Int
end

function (::MutualInformation)(x, y; kwargs...)
    return get_mutual_information(x, y; estimator="shrinkage")
end

function (::NormalisedMutualInformation)(x, y; kwargs...)
    # https://en.wikipedia.org/wiki/Mutual_information#Normalized_variants
    mi = (MutualInformation())(x, y)
    return mi / sqrt(
        get_entropy(x; estimator="shrinkage") * get_entropy(y; estimator="shrinkage")
    )
end

function (::ConditionalMutualInformation)(x, y; conditioned_variable, kwargs...)
    return get_conditional_mutual_information(
        x, y, conditioned_variable; estimator="shrinkage"
    )
end

function (::PearsonCorrelation)(x, y; kwargs...)
    return cor(x, y)
end

function (::SpearmanCorrelation)(x, y; kwargs...)
    return corspearman(x, y)
end

"""
Calculate feature relevance between x and y, handling missing and cases where
there is only one unique value.
kwargs are passed through to criterion.
"""
function calculate_feature_stats(
    criterion::FeatureRelevanceCriterion,
    x::Union{AbstractVector{Union{Missing,Float64}},AbstractVector{Float64}},
    y::Union{AbstractVector{Union{Missing,Float64}},AbstractVector{Float64}};
    conditioned_variable::Union{
        AbstractVector{Union{Missing,Float64}},AbstractVector{Float64}
    }=Float64[],
    kwargs...,
)
    # NOTE: relaxing the above type signature from Float64 to T<:Real works in Julia 1.2
    # but causes the GreedyJMI method to segfault in Julia 1.1 in some instances. So
    # keeping it as Float64 for now. There is a test for this in test/feature_selection.jl

    tmp_df = missing

    # Need to ensure we match up non-missing indices between all variables
    non_na_idx = intersect(
        findall(.!ismissing.(x) .& isfinite.(x)), findall(.!ismissing.(y) .& isfinite.(y))
    )
    if !isempty(conditioned_variable)
        non_na_idx = intersect(
            non_na_idx,
            findall(.!ismissing.(conditioned_variable) .& isfinite.(conditioned_variable)),
        )
    end

    length(non_na_idx) == 0 && return tmp_df

    x = convert(Vector{Float64}, x[non_na_idx])
    y = convert(Vector{Float64}, y[non_na_idx])
    if !isempty(conditioned_variable)
        conditioned_variable = convert(Vector{Float64}, conditioned_variable[non_na_idx])
    end

    y_percent_diff = (maximum(y) - minimum(y)) ./ minimum(y)
    x_percent_diff = (maximum(x) - minimum(x)) ./ minimum(x)

    # check if more than one discretized value
    # take absolute value since julia has -0.0 and 0.0 and these are the same,
    # but don't get combined with unique
    if (length(unique(abs.(x))) > 1) &
       (length(unique(abs.(y))) > 1) &
       # make sure that there aren't just different values due to rounding differences
       (abs(x_percent_diff) > 1e-5) &
       (abs(y_percent_diff) > 1e-5)
        relevance = criterion(x, y; conditioned_variable=conditioned_variable, kwargs...)
        tmp_df = DataFrame(; relevance=relevance)
    end

    return tmp_df
end

"""
Calculate feature relevance pairwise between all colummns of x and y.
"""
function calculate_feature_stats(
    criterion::FeatureRelevanceCriterion, x::DataFrame, y::DataFrame; kwargs...
)
    feature_stats = DataFrame(; relevance=Float64[], x_name=Symbol[], y_name=Symbol[])

    x_features = names(x)
    y_features = names(y)

    for y_feature in y_features
        for x_feature in x_features
            tmp_df = calculate_feature_stats(
                criterion, x[:, x_feature], y[:, y_feature]; kwargs...
            )
            if !ismissing(tmp_df)
                tmp_df[!, :x_name] .= Symbol(x_feature)
                tmp_df[!, :y_name] .= Symbol(y_feature)
                append!(feature_stats, tmp_df)
            end
        end
    end
    return feature_stats
end

function _safety_check(target::DataFrame, features::DataFrame)
    # We only allow one target column
    if ncol(target) != 1
        err = "target has $(ncol(target)) columns, only 1 allowed"
        throw(ArgumentError(err))
    end
    # Safety check: we should never have a target as a feature
    # Here we just check the column names
    cheating_features = intersect(names(features), names(target))
    if !isempty(cheating_features)
        err = "Features overlap with target! " * "Intersection = $cheating_features"
        throw(ArgumentError(err))
    end
end

"""
    _select_single_target(method::FeatureSelectionMethod, target::DataFrame, features::DataFrame)
    -> (selected_features, scores)

Use a `FeatureSelectionMethod` to select columns from `features` which are most informative
about `target`.

Available `FeatureSelectionMethod`s are: `Top`, `GreedyMRMR`, `GreedyJMI`.

## Arguments:
- `method`: How to select the features.
- `target`: A DataFrame containing a single target column.
- `features`: A DataFrame containing feature columns.

## Returns:
- `selected_features`: Vector of feature names selected.
- `scores`: Scores associated with these features.
"""
function _select_single_target(top::Top, target::DataFrame, features::DataFrame)
    _safety_check(target, features)

    # If N is too large, just return all features
    num_features = length(names(features))
    num_requested = top.N
    if num_requested >= num_features
        @warn "Requested $(num_requested) features out of $(num_features), " *
              "returning all features"
        num_requested = num_features
    end

    feature_stats = calculate_feature_stats(top.criterion, features, target)
    # Select top N features
    sort!(feature_stats, :relevance; rev=true)
    selected_features = feature_stats[1:num_requested, :x_name]
    scores = feature_stats[1:num_requested, :relevance]

    @assert selected_features == unique(selected_features)
    @assert length(selected_features) == num_requested
    return DataFrame(:feature => selected_features, :score => scores)
end

function _select_single_target(
    rfmetric::RandomForestMethod, target::DataFrame, features::DataFrame
)
    _safety_check(target, features)

    y = target[:, 1]

    # replace missings with NaN
    x = Matrix{Float64}(coalesce.(features, NaN))

    return DataFrame(:feature => Symbol.(names(features)), :score => rfmetric(x, y))
end

_use_β(::GreedyMRMR) = true
_use_γ(::GreedyMRMR) = false

_use_β(::GreedyJMI) = true
_use_γ(::GreedyJMI) = true

"""
    _select_single_target(
        greedy::GreedyMethod,
        target::DataFrame,
        features::DataFrame
    ) -> (selected_features, scores)

Select according to a greedy strategy.
This is based on [1] equation 17/18.

[1] Brown et al., 2012. Conditional Likelihood Maximisation: A Unifying Framework for
Information Theoretic Feature Selection, JMLR 13.
"""
function _select_single_target(greedy::GreedyMethod, target::DataFrame, features::DataFrame)
    mi = MutualInformation()
    cmi = ConditionalMutualInformation()

    _safety_check(target, features)

    features_selected = Symbol[]
    scores = []
    # Features left to choose from
    features_remaining = Symbol.(deepcopy(names(features)))

    # Relevance of all features to this target. This is target-specific. There is no
    # way around computing this for all features, so just compute it now.
    # This is the first term in the RHS of [1] eq. 18
    target_relevance = calculate_feature_stats(mi, features, target)

    # Relevance of features to each other - fill in as necessary.
    # This is independent of the target.
    # This is the second term in the RHS of [1] eq. 18
    feature_relevance = DataFrame()

    # Relevance of features to each other given target - fill in as necessary.
    # This is target-specific.
    # This is the third term in the RHS of [1] eq. 18
    feature_conditional_relevance = DataFrame()

    # Select N features greedily
    for f in 1:(greedy.N)
        if isempty(features_remaining)
            @warn "Ran out of features, returning all"
            break
        end

        # Relevancy of all *remaining* features to this target
        stats = filter(x -> x[:x_name] ∈ features_remaining, target_relevance)

        if isempty(features_selected)
            # Just pick best feature
            sort!(stats, :relevance; rev=true)
            new_feature = stats[1, :x_name]
            push!(features_selected, new_feature)
            push!(scores, stats[1, :relevance])
            # Remove from features under consideration
            filter!(x -> x ≠ new_feature, features_remaining)
            continue
        end

        # Calculate redundancy of all features remaining, i.e. relevance to features
        # already selected

        if _use_β(greedy)
            # This choice of β corresponds to MRMR from [1]
            β = 1.0 / length(features_selected)
            # Calculate MI between all features remaining and most recently
            # selected feature.
            # Keep track of this to avoid re-computation.
            df = calculate_feature_stats(
                mi, features[:, features_remaining], features[:, features_selected[[end]]]
            )
            feature_relevance = vcat(feature_relevance, df)
            # Consolidate for all features remaining
            #sum_fr = by(feature_relevance, :x_name, redundancy_β = :relevance => sum)
            sum_fr = combine(
                groupby(feature_relevance, :x_name), :relevance => sum => :redundancy_β
            )

            sum_fr[!, :redundancy_β] = β * sum_fr[:, :redundancy_β]
            stats = innerjoin(stats, sum_fr[:, [:x_name, :redundancy_β]]; on=:x_name)
        else
            stats[!, :redundancy_β] .= 0.0
        end

        if _use_γ(greedy)
            # This choice of γ along with the β above corresponds to JMI from [1]
            γ = 1.0 / length(features_selected)
            # Calculate CMI between all features remaining and most recently
            # last selected feature.
            # Keep track of this to avoid re-computation.
            df = calculate_feature_stats(
                cmi,
                features[:, features_remaining],
                features[:, features_selected[[end]]];
                conditioned_variable=target[:, 1],
            )
            feature_conditional_relevance = vcat(feature_conditional_relevance, df)
            # Consolidate for all features remaining
            # sum_fcr = by(
            #     feature_conditional_relevance,
            #     :x_name,
            #     redundancy_γ = :relevance => sum
            # )
            sum_fcr = combine(
                groupby(feature_conditional_relevance, :x_name),
                :relevance => sum => :redundancy_γ,
            )
            sum_fcr[!, :redundancy_γ] = γ * sum_fcr[:, :redundancy_γ]
            stats = innerjoin(stats, sum_fcr[:, [:x_name, :redundancy_γ]]; on=:x_name)
        else
            stats[!, :redundancy_γ] .= 0.0
        end
        stats.score = stats.relevance .- (stats.redundancy_β .- stats.redundancy_γ)
        sort!(stats, :score; rev=true)

        # Select top feature
        new_feature = stats[1, :x_name]
        push!(features_selected, new_feature)
        push!(scores, stats[1, :score])
        # Remove from features under consideration
        filter!(x -> x ≠ new_feature, features_remaining)
    end
    @assert features_selected == unique(features_selected)
    return DataFrame(:feature => features_selected, :score => scores)
end

"""
    select(
        fsm::FeatureSelectionMethod,
        targets::DataFrame,
        features::DataFrame
    ) -> feature_mapping::Dict{Symbol, DataFrame}

For each column in `targets`, use method `fsm` to select features from among the columns
of `features`.

## Returns:
- Dictionary `feature_mapping`. The keys are the column names from `targets`, and the
  values are DataFrames with columns `:feature` and `:score`. The features are in selection
  order.
"""
function select(fsm::FeatureSelectionMethod, targets::DataFrame, features::DataFrame)
    # For each target node, get features
    feature_mapping = Dict{Symbol,DataFrame}()
    for target_name in Symbol.(names(targets))
        feature_mapping[target_name] = _select_single_target(
            fsm, targets[:, [target_name]], features
        )
    end
    return feature_mapping
end
