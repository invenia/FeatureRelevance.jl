"""
    Algorithm

The algorithm to use when evaluating feature relevancy.
New algorithms must implement `selection(alg, features, target)`
"""
abstract type Algorithm end

"""
    selection(algorithm, features, target) -> (idx, scores)

Use the `algorithm` to find the most relevant `features` for the `target`.
Returning the index and relevance score for each selected feature.

Available algorithms include: `Top`, `GreedyMRMR`, `GreedyJMI`.

# Arguments:
- `algorithm`: How to select the features.
- `features`: An iterable of feature vectors to consider
- `target`: An abstract vector observations for a single target

# Returns:
- `idx`: Location of each selected feature in features.
- `n_obs`: Number of feature-target observations compared
- `scores`: Scores associated with these features.
"""
selection(alg, features, target)

"""
    Top(; criterion=MutualInformation(), n=0)
    Top(f; n=0)

Select the top n relevant features.
This is fast but might result in redundant features.
Mutual information maximisation (MIM).

# Arguments
- `criterion`: The function or criteria to use for evaluating feature relevance
- `n`: A positive values indicates the number of most relevant features to select.
       Non-positive values indicate the number of least relevant features to drop.
       A value of zero indicates that all features should be returned regardless of score.
- `f`: Passing a function as the first positional argument support do-block syntax for
  custom criterion.
"""
Base.@kwdef struct Top{T<:Union{Criterion,Function}} <: Algorithm
    criterion::T=MutualInformation()
    n::Int=0
end

# Utility constructor for do-block syntax
Top(f; kwargs...) = Top(; criterion=f, kwargs...)

function selection(alg::Top, features, target)
    criterion = alg.criterion

    nfeatures = length(features)
    n = if alg.n <= 0
        nfeatures + alg.n
    elseif 0 < alg.n <= nfeatures
        alg.n
    else
        @debug("Requested $(alg.n) out of $nfeatures features, returning all.")
        nfeatures
    end

    # Calculate our relevance stats
    # this returns a Vector{Tuple{Int64, Float64}}
    # The first item in the tuple is the number of elements compared
    # (ignoring missing values, etc).  The second is the relevance score
    stats = [relevance(criterion, target, f) for f in features]

    n_obs, scores = first.(stats), last.(stats)

    # Get the sorted order
    sorted = sortperm(scores; rev=true)

    # Drop any missing stats
    filtered = Iterators.filter(i -> !ismissing(stats[i]), sorted)

    # Take at most n of the highest values
    idx = collect(Iterators.take(filtered, n))

    length(idx) == n || @debug "Too few features selected ($(length(idx))), expected $n. Probably due to the relevance score being missing for certain feature target paris."
    return idx, n_obs[idx], scores[idx]
end

"""
    Greedy(; n=0, β=true, γ=false, positive=false)

Select according to a greedy strategy. This is based on [1] equation 17/18.

# Arguments
- `n`: Number of relevant features to select
- `β`: Whether to use `β` to MRMR from [1]
- `γ`: Whether to use `γ` to JMI from [1]
- `positive`: Only return positive scores (ie: score > redundancy)

[1] Brown et al., 2012. Conditional Likelihood Maximisation: A Unifying Framework for
Information Theoretic Feature Selection, JMLR 13.
"""
Base.@kwdef struct Greedy <: Algorithm
    n::Int=0
    β::Bool=true
    γ::Bool=false
    positive::Bool=false
end

"""
    GreedyMRMR(; n=0, positive=false)

Greedy selection taking into account pairwise dependence of features, but assuming pairwise
class-conditional independence.
Maximum relevancy minimum redundancy (MRMR).
"""
GreedyMRMR(; n=0, positive=false) = Greedy(; n=n, β=true, γ=false, positive=positive)

"""
    GreedyJMI(; n=0, positive=false)

Greedy selection taking into account pairwise dependence of features, and also taking into
account pairwise class-condtional dependence.
Joint mutual information (JMI).
"""
GreedyJMI(; n=0, kwargs...) = Greedy(; n=n, β=true, γ=true, kwargs...)

function selection(alg::Greedy, features, target)
    # Since we're gonna need to index into each feature repeatedly we collect the feature
    # vectors (vector of vectors)
    X = collect(features)
    if alg.n > length(X)
        @debug("Requested $(alg.n) out of $(length(X)) features, returning all.")
    end

    mi = MutualInformation()
    cmi = ConditionalMutualInformation()

    # We're gonna need to reference all of these relevances so we'll just pre-compute them
    indices = collect(eachindex(X))
    # below returns a Vector{Tuple{Int64, Float64}}
    # The first item in the tuple is the number of elements compared
    # (ignoring missing values, etc).  The second is the relevance score
    stats = [relevance(mi, target, f) for f in X]

     n_obs, scores = first.(stats), last.(stats)

    # Running sum of our feature relevances, conditioned and unconditioned on the target
    β_rel, γ_rel = (zeros(length(relevances)) for _ in 1:2)

    # Note any features that have a `missing` relevance and remove those indices
    # from consideration.
    mask = map(ismissing, relevances)
    deleteat!(indices, mask)
    nfeatures = count(!, mask)

    # Determine the number of selected indices and scores we're returning
    n = if alg.n <= 0
        nfeatures + alg.n
    elseif 0 < alg.n <= nfeatures
        alg.n
    else
        @warn("Too many relevance scores are missing. Returning $n.")
        nfeatures
    end

    # Pre-allocate our selected items
    # NOTE: There might be a cleaner iteration algorithm, but reusing pre-allocated
    # redundancy arrays seems important for performance.
    selected_indices = Vector{Int}(undef, n)
    selected_scores = Vector{nonmissingtype(eltype(relevances))}(undef, n)
    for i in 0:n-1
        remaining_indices = setdiff(indices, selected_indices[1:i])
        remaining_scores = relevances[remaining_indices]

        # Only compute redundancy if this isn't our first iteration
        if i > 0
            prev = X[selected_indices[i]]

            for (j, k) in enumerate(remaining_indices)
                redundancy = 0.0

                if alg.β
                    β_rel[k] += last(relevance(mi, prev, X[k]))
                    redundancy = (1.0 / i) * β_rel[k]
                end

                if alg.γ
                    γ_rel[k] += relevance(cmi, prev, X[k], target)[2]
                    redundancy -= (1.0 / i) * γ_rel[k]
                end

                remaining_scores[j] -= redundancy
            end
        end

        x, idx = findmax(remaining_scores)
        selected_indices[i + 1] = remaining_indices[idx]
        selected_scores[i + 1] = x
    end

    if alg.positive
        pos_idx = findall(>(0.0), selected_scores)
        if length(pos_idx) < length(selected_scores)
            return selected_indices[pos_idx], n_obs[selected_indices[pos_idx]], selected_scores[pos_idx]
        end
    end

    return selected_indices, n_obs[selected_indices], selected_scores
end
