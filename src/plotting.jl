function binned_df(xs, ys, nbins)
    # compute the bin indices
    quantiles = 0.0:1.0/nbins:1.0
    edges = quantile(sort(xs), quantiles)
    edges = unique(edges) # in case there are repeated values and some edges are the same
    nbins_left = length(edges) - 1 # number of bins left after removing zero width bins
    bin_indices = [findlast(edge -> edge <= x, edges) for x in xs]
    # push the highest value from overflow bin to last bin
    bin_indices[findall(==(nbins_left+1), bin_indices)] .= nbins_left

    # compute the means and stds per bin
    df = DataFrame(; xs, ys, bin_indices)
    gb = groupby(df, :bin_indices)
    bin_df = combine(gb, :ys => mean, :ys => std)
    bin_df.xs = 0.5*(edges[2:end] + edges[1:end-1]) # centres

    return bin_df
end

"""
    binnedmeanandstd(xs, ys; bins=10, ylim=:auto)

Equally split the `xs` and `ys` values into `bins`, based on the values of `xs`.
Plots the values of `ys` in each bin, along with the mean and standard deviation.

Additionally, it plots a gray band where the same calculation is done on the shuffled
dataset, which allows us to eyeball the level of fluctuations due to finite sample size.

Arguments
---
- xs: AbstractVector of `xs` (the variable in which the binning is done)
- ys: AbstractVector of `ys` (the variable over which the mean and std are computed)
- bins: number of bins
- ylim: y-range for showing the data
"""
@userplot BinnedMeanAndStd

# NB: bins not nbins (https://github.com/JuliaPlots/RecipesBase.jl/issues/86)
@recipe function f(b::BinnedMeanAndStd; bins=10, ylim=:auto)

    # check and extract args
    if length(b.args) != 2 || !(typeof(b.args[1]) <: AbstractVector) ||
        !(typeof(b.args[2]) <: AbstractVector)
        error("binnedmeanandstd should be given two vectors. Got: $(typeof(bargs))")
    end
    xs, ys = b.args

    # bin the data
    bin_df = binned_df(xs, ys, bins)
    bin_df_shuffle = binned_df(xs, shuffle(ys), bins)

    # common settings
    ymin = quantile(bin_df.ys_mean - bin_df.ys_std, 0.1)
    ymax = quantile(bin_df.ys_mean + bin_df.ys_std, 0.9)
    yrange = ymax - ymin
    ylim := (ymin - 0.4 * yrange, ymax + 0.4 * yrange)
    legend := true
    ylabel := "price"
    xlabel := "feature"

    # data
    @series begin
        label := "data"
        markercolor := :orange
        alpha := 0.1
        seriestype := :scatter
        xs, ys
    end

    # real data
    @series begin
        label := "mean"
        seriestype := :line
        linecolor := :orange
        linewidth := 3
        alpha := 1.0
        bin_df.xs, bin_df.ys_mean
    end
    @series begin
        label := "std"
        c := :orange
        seriestype := :line
        fillrange := bin_df.ys_mean + bin_df.ys_std
        alpha := 0.35
        bin_df.xs, bin_df.ys_mean - bin_df.ys_std
    end

    # shuffled data
    @series begin
        label := "shuffled"
        seriestype := :line
        linecolor := :gray
        alpha := 1.0
        bin_df_shuffle.xs, bin_df_shuffle.ys_mean
    end
    @series begin
        c := :gray
        label := nothing
        seriestype := :line
        fillrange := bin_df_shuffle.ys_mean + bin_df_shuffle.ys_std
        alpha := 0.35
        bin_df_shuffle.xs, bin_df_shuffle.ys_mean - bin_df_shuffle.ys_std
    end
    return nothing
end

"""
    convergence(f, features, targets; n=100, sz=Int[], group_names=String[], contiguous=false)

Re-calculate the feature relevance between each feature and target with various sample sizes.
Useful for checking that our criterion metric is converging for various feature x target pairs as our sample/lookback size grows.
You'll find that some feature x target pairs converge more easily than others based on their relative complexity.
However, if scores aren't converging then this may indicate an issue with your desired metric.
By default this will consider samples of `0.05:0.05:1.0 * nobs` using random sampling w/o replacement.

1. Colour is a logic feature grouping (temp, wind, load, etc)
2. X-axis is the varying sample sizes
3. Y-axis is the computed relevance

# Arguments
- `f``: A callable function/criterion for calculating relevance between individual targets and features
  Individual scores are calculated columnwise, similar to `report` and `selection`.
- `features`: iterable of tables representing logical groupings of features (e.g., temperature, wind, load, prices).
  If only a single table/matrix is provided then each individual column is treated as its own group.
- `targets`: table of all target values

# Keywords
- `n`: Number of samples to draw for each size (adjusts smoothness)
- `sz`: A range of dataset sizes to consider (adjusts shape)
- `group_names`: The name of each feature group. Should have the same length as the number of tables in `features`
- `contiguous`: Whether our drawn samples must be for a contiguous window (e.g., daily lookback)
"""
@userplot Convergence

@recipe function f(c::Convergence; n=100, sz=[], group_names=[], contiguous=false)
    func, features, targets = c.args
    #=
    `typejoin`/`eltype` may return too wide of a type to identify a table correctly,
    so we check each group/component separately.

    Example:
    ```
    julia> @show eltype(f1) Tables.istable(eltype(f1)) all(T -> Tables.istable(T), typeof.(f1))
    eltype(f1) = NamedTuple
    Tables.istable(eltype(f1)) = false
    all((T->begin
                #= REPL[30]:1 =#
                Tables.istable(T)
            end), typeof.(f1)) = true
    true

    julia> @show eltype(f2) Tables.istable(eltype(f2)) all(T -> Tables.istable(T), typeof.(f2))
    eltype(f2) = NamedTuple{names, Tuple{Vector{Float64}, Vector{Float64}}} where names
    Tables.istable(eltype(f2)) = true
    all((T->begin
                #= REPL[31]:1 =#
                Tables.istable(T)
            end), typeof.(f2)) = true
    true
    ```
    =#
    issingular = !all(x -> Tables.istable(x) || x isa AbstractArray, features)
    features = issingular ? _get_columns(features) : features
    gnames = isempty(group_names) ? (1:length(features)) : group_names
    sampler = contiguous ? contiguoussampled : randomsampled

    nnames = length(gnames)
    nfeatures = length(features)
    if nnames != nfeatures
        throw(ArgumentError(
            "Number of groups in `features` and `group_names` do not match: " *
            "($nfeatures, $nnames)"
        ))
    end

    legend := true
    ylabel := "score"
    xlabel := "sample size (contiguous=$contiguous)"

    for (label, group) in zip(gnames, features)
        iter = Iterators.product(_get_columns(targets), _get_columns(group))
        nobs = length(first(first(iter)))
        x = unique(filter(>(1), isempty(sz) ? floor.(Int, 0.05:0.05:1.0*nobs) : sz))
        # Preallocate a matrix of our nsamples * npairs + npairs (for NaNs)
        # Insert NaNs to split each independent line
        # https://docs.juliaplots.org/latest/input_data/#Unconnected-Data-within-same-groups
        nsamples = length(x)
        npairs = length(iter)
        nrow = nsamples * npairs + npairs
        results = fill(NaN, (nrow, 2))

        Threads.@threads for (i, t) in collect(enumerate(iter))
            feature, target = t
            start_idx = (i - 1) * nsamples + i
            end_idx = start_idx + nsamples - 1
            idx = start_idx:end_idx
            # @show nsamples idx x
            results[idx, 1] = float.(x)
            results[idx, 2] = [sampler(func, feature, target; n=n, sz=s) for s in x]
        end

        @series begin
            label := label
            seriestype := :path
            results[:, 1], results[:, 2]
        end
    end

    return nothing
end

function randomsampled(f, x, y; n, sz)
    return map(1:n) do _
        idx = sample(1:length(x), sz; replace=false, ordered=true)
        f(x[idx], y[idx])
    end |> median
end

function contiguoussampled(f, x, y; n, sz)
    # NOTE: As our window size grows we'll be able to draw fewer unique samples
    iter = unique(length(x) > sz ? sample(1:length(x)-sz, n) : [1])

    return map(iter) do start_idx
        idx = start_idx:start_idx+sz-1
        f(x[idx], y[idx])
    end |> median
end
