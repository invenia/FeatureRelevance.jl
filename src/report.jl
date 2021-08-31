"""
    report([criterion,] alg, targets, features)

For each column in `targets`, use method `alg` to select features from among the columns
of `features`.

# Arguments
- `criterion`: Optional criterion to use with the provide `alg` (not supported by some algs)
- `alg`: The algorithm to use for selecting relevant `features` for each `target`
- `targets`: A table or matrix for 1 or more target values
- `features`: A table or matrix for 1 or more features

## Returns:
- A `Tables.rowtable` of each selected target x feature relevance score with columns
  `:target`, `:feature` and `:score`.
"""
report(alg::Algorithm, targets, features) = _report(features, targets, alg)
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

    results = pmap(enumerate(_get_columns(y))) do (i, target)
        selected = selection(args..., target, _get_columns(X))
        result = map(zip(selected...)) do (j, score)
            return (; target=target_names[i], feature=feature_names[j], score=score)
        end
    end

    return collect(Iterators.flatten(results))
end

# Some utility functions for differentiating different inputs
_validate(data) = data
_validate(data::AbstractVector) = reshape(data, (length(data), 1))
_get_names(data) = Tables.istable(data) ? Tables.columnnames(data) : 1:size(data, 2)
_get_columns(data) = Tables.istable(data) ? Tables.columns(data) : eachcol(data)
