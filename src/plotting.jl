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

