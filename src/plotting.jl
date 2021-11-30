function binned_df(xs, ys, nbins)
    # compute the bin indices
    quantiles = 0.0:1.0/nbins:1.0
    edges = quantile(sort(xs), quantiles)
    bin_indices = [findlast(edge -> edge <= x, edges) for x in xs]
    # push the highest value from overflow to last bin
    bin_indices[findfirst(==(nbins+1), bin_indices)] = nbins

    # compute the means and stds per bin
    df = DataFrame(; xs, ys, bin_indices)
    gb = groupby(df, :bin_indices)
    bin_df = combine(gb, :ys => mean, :ys => std)
    bin_df.xs = 0.5*(edges[2:end] + edges[1:end-1]) # centres
    
    return bin_df
end

@userplot VisualiseFeature

# NB: bins not nbins (https://github.com/JuliaPlots/RecipesBase.jl/issues/86)
@recipe function f(b::VisualiseFeature; bins=10)

    # check and extract args
    if length(b.args) != 2 || !(typeof(b.args[1]) <: AbstractVector) ||
        !(typeof(b.args[2]) <: AbstractVector)
        error("visualisefeature should be given two vectors. Got: $(typeof(bargs))")
    end
    xs, ys = b.args

    # bin the data
    bin_df = binned_df(xs, ys, bins)
    bin_df_shuffle = binned_df(xs, shuffle(ys), bins)

    # common settings
    legend := true
    ymin = quantile(bin_df.ys_mean - bin_df.ys_std, 0.1)
    ymax = quantile(bin_df.ys_mean + bin_df.ys_std, 0.9)
    yrange = ymax - ymin
    ylim := (ymin - 0.4 * yrange, ymax + 0.4 * yrange)
    ylabel --> "price"
    xlabel --> "feature"

    # data
    @series begin
        label := "data"
        markercolor := :orange
        alpha := 0.2
        seriestype := :scatter
        xs, ys
    end

    # real data
    @series begin
        label := "mean"
        seriestype := :line
        linecolor := :orange
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
end

