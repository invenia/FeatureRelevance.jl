"""
    report(alg, targets, features)
    report(criterion, targets, features)
    report(criterion, alg, targets, features)

For each column in `targets`, use method `alg` to select features from among the columns
of `features`.

# Arguments
- `criterion`: Optional criterion to use with the provide `alg` (not supported by some algs)
- `alg`: The algorithm to use for selecting relevant `features` for each `target`.
         If only `criterion` is specified then this assumed to be `Top(Inf)`.
- `targets`: A table or matrix for 1 or more target values
- `features`: A table or matrix for 1 or more features

## Returns:
- A `Tables.rowtable` of each selected target x feature relevance score with columns
  `:target`, `:feature` and `:score`.
"""
report(alg::Algorithm, targets, features) = _report(features, targets, alg)
report(criterion::Criterion, targets, features) = report(criterion, ALL, targets, features)
function report(criterion, alg::Algorithm, targets, features)
    return _report(features, targets, criterion, alg)
end

function _report(features, targets, args...)
    X, y = _validate(features), _validate(targets)

    feature_names = _get_names(X)
    target_names = _get_names(y)

    result_names = (:target, :feature, :score)
    result_types = Tuple{eltype(target_names),eltype(feature_names),Float64}
    T = NamedTuple{result_names,result_types}

    # We use a Channel as a buffer since we don't know how many results to expect.
    results = Channel(; ctype=T, csize=256) do ch
        Threads.@threads for (i, target) in collect(enumerate(_get_columns(y)))
            selected = selection(args..., target, _get_columns(X))
            for (j, score) in zip(selected...)
                record = (; target=target_names[i], feature=feature_names[j], score=score)

                put!(ch, record)
            end
        end
    end

    return collect(results)
end
