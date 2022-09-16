"""
    relevance(criterion, x, y, z...) -> score

A wrapper function around a criterion function/callable which handles
dropping missing values and ensure there is more than 1 unique discrete value.

# Arguments
- `criterion`: The function or callable criterion metrics to apply to the other arguments.
- `x`: The first set of values to compare
- `y`: The second set of values to compare
- `z`: An optional set of values for conditioning the criterion on.

# Returns
- The number of elements compared.  This is important for interpretation since
  very high relevance scores may be observed when there are many missing values
- The relevance score as defined by the `criterion`
"""
function relevance(criterion, x, y, z...)
    mask = map(t -> all(v -> !ismissing(v) && isfinite(v), t), zip(x, y, z...))

    # Early exit if we've filtered out all values
    iszero(count(mask)) && return 0, missing

    # make sure the values are actually different
    args = map((x, y, z...)) do arg
        v = disallowmissing(arg[mask])
        percent_diff = (maximum(v) - minimum(v)) ./ minimum(v)
        length(unique(abs.(v))) > 1 && abs(percent_diff) > 1e-5 && return v
        return missing
    end
    any(ismissing, args) && return 0, missing

    return count(mask), criterion(args...)
end
