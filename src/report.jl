"""
    report(alg, features, targets)

For each column in `targets`, use method `alg` to select features from among the columns
of `features`.

# Arguments
- `alg`: The algorithm to use for selecting relevant `features` for each `target`.
- `targets`: A table or matrix for 1 or more target values
- `features`: A table or matrix for 1 or more features

## Returns:
- A `Tables.rowtable` of each selected target x feature relevance score with columns
  `:target`, `:feature`, `:n_obs` and `:score`.
"""
function report(alg::Algorithm, features, targets)
    X, y = _validate(features), _validate(targets)

    feature_names = _get_names(X)
    target_names = _get_names(y)

    result_names = (:target, :feature, :n_obs, :score)
    result_types = Tuple{eltype(target_names),eltype(feature_names),Int64,Float64}
    T = NamedTuple{result_names,result_types}

    # We use a Channel as a buffer since we don't know how many results to expect.
    results = Channel(; ctype=T, csize=256) do ch
        Threads.@threads for (i, target) in collect(enumerate(_get_columns(y)))
            selected = selection(alg, _get_columns(X), target)
            for (j, n_obs, score) in zip(selected...)
                record = (; target=target_names[i], feature=feature_names[j], n_obs = n_obs, score=score)

                put!(ch, record)
            end
        end
    end

    return collect(results)
end
