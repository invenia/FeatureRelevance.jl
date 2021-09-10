"""
    Algorithm

The algorithm to use when evaluating feature relevancy.
New algorithms must implement `selection(alg, target, features...)`
"""
abstract type Algorithm end

"""
    selection(algorithm, target, features) -> (idx, scores)

Use the `algorithm` to find the most relevant `features` for the `target`.
Returning the index and relevance score for each selected feature.

Available algorithms include: `Top`, `GreedyMRMR`, `GreedyJMI`.

# Arguments:
- `algorithm`: How to select the features.
- `target`: An abstract vector observations for a single target
- `features`: An iterable of feature vectors to consider

# Returns:
- `idx`: Location of each selected feature in features.
- `scores`: Scores associated with these features.
"""
selection(alg, target, features...)

"""
    Top(; n)

Select the top n relevant features.
This is fast but might result in redundant features.
Mutual information maximisation (MIM).

# Arguments
- `n`: Number of most relevant features to select
"""
Base.@kwdef struct Top <: Algorithm
    n::Int
end

selection(top::Top, args...) = selection(MutualInformation(), top, args...)

function selection(criterion, alg::Top, target, features)
    # If N is too large, just return all features
    alg.n < length(features) ||
        @warn("Requested $(alg.n) out of $(length(features)) features, returning all.")

    n = min(alg.n, length(features))

    # Calculate our relevance stats
    stats = [relevance(criterion, target, f) for f in features]

    # Get the sorted order
    sorted = sortperm(stats; rev=true)

    # Drop any missing stats
    filtered = Iterators.filter(i -> !ismissing(stats[i]), sorted)

    # Take at most n of the highest values
    idx = collect(Iterators.take(filtered, n))

    @assert length(idx) == n "Too few features selected ($(length(idx))), expected $n."
    return idx, stats[idx]
end

"""
    Greedy(; n, β=true, γ=false)

Select according to a greedy strategy. This is based on [1] equation 17/18.

# Arguments
- `n`: Number of relevant features to select
- `β`: Whether to use `β` to MRMR from [1]
- `γ`: Whether to use `γ` to JMI from [1]

[1] Brown et al., 2012. Conditional Likelihood Maximisation: A Unifying Framework for
Information Theoretic Feature Selection, JMLR 13.
"""
Base.@kwdef struct Greedy <: Algorithm
    n::Int
    β::Bool = true
    γ::Bool = false
end

"""
    GreedyMRMR(; n)

Greedy selection taking into account pairwise dependence of features, but assuming pairwise
class-condtional independence.
Maximum relevancy minimum redundancy (MRMR).
"""
GreedyMRMR(n) = Greedy(; n=n, β=true, γ=false)

"""
    GreedyJMI(; n)

Greedy selection taking into account pairwise dependence of features, and also taking into
account pairwise class-condtional dependence.
Joint mutual information (JMI).
"""
GreedyJMI(n) = Greedy(; n=n, β=true, γ=true)

function selection(alg::Greedy, target, features)
    # Since we're gonna need to index into each feature repeatedly we collect the feature
    # vectors (vector of vectors)
    X = collect(features)
    alg.n < length(X) ||
        @warn("Requested $(alg.n) out of $(length(X)) features, returning all.")

    mi = MutualInformation()
    cmi = ConditionalMutualInformation()

    # We're gonna need to reference all of these relevances so we'll just pre-compute them
    indices = collect(eachindex(X))
    relevances = [relevance(mi, target, f) for f in X]

    # Running sum of our feature relevances, conditioned and unconditioned on the target
    β_rel, γ_rel = (zeros(length(relevances)) for _ in 1:2)

    # Note any features that have a `missing` relevance and remove those indices
    # from consideration.
    mask = map(ismissing, relevances)
    deleteat!(indices, mask)

    # Determine the number of selected indices and scores we're returning
    n = min(alg.n, count(!, mask))
    alg.n <= count(!, mask) || @warn("Too many relevance scores are missing. Returning $n.")

    # Pre-allocate our selected items
    # NOTE: There might be a cleaner iteration algorithm, but reusing pre-allocated
    # redundancy arrays seems important for performance.
    selected_indices = Vector{Int}(undef, n)
    selected_scores = Vector{nonmissingtype(eltype(relevances))}(undef, n)
    i = 0
    while i < n
        remaining_indices = setdiff(indices, selected_indices[1:i])
        remaining_scores = relevances[remaining_indices]

        # Only compute redundancy if this isn't our first iteration
        if i > 0
            prev = X[selected_indices[i]]

            for (j, k) in enumerate(remaining_indices)
                redundancy = 0.0

                if alg.β
                    β_rel[k] += relevance(mi, prev, X[k])
                    redundancy = (1.0 / i) * β_rel[k]
                end

                if alg.γ
                    γ_rel[k] += relevance(cmi, prev, X[k], target)
                    redundancy -= (1.0 / i) * γ_rel[k]
                end

                remaining_scores[j] -= redundancy
            end
        end

        x, idx = findmax(remaining_scores)
        selected_indices[i + 1] = remaining_indices[idx]
        selected_scores[i + 1] = x
        i += 1
    end

    return selected_indices, selected_scores
end
