@testset "BinnedMeanAndStd" begin
    @testset "nice case" begin
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

    @testset "repeated highest values" begin
        xs = vcat(rand(98), [1.2, 1.2])
        ys = randn(100)
        result = binnedmeanandstd(xs, ys)
        @test length(result.subplots[1].series_list[1].plotattributes[:x]) == 100
    end

    @testset "multiple zero-width bins" begin
        xs = vcat(rand(5), fill(rand(), 95))
        ys = randn(100)
        result = binnedmeanandstd(xs, ys)
        @test length(result.subplots[1].series_list[2].plotattributes[:x]) == 2 # 2 bins only
    end
end
