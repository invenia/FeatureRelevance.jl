@testset "BinnedMeanAndStd" begin
    ylabel = "my ylabel"
    bins = 12
    ylim = (-2, 2)
    result = binnedmeanandstd(randn(100), randn(100); ylabel, bins, ylim)

    @test length(result.subplots[1].series_list) === 5 # data, 2x mean, 2x std
    @test result.subplots[1][:xaxis][:guide] == "feature"
    @test result.subplots[1][:yaxis][:guide] == ylabel
    @test length(result.subplots[1].series_list[2].plotattributes[:x]) == bins
    @test result.subplots[1].attr[:yaxis][:lims] == ylim
end
