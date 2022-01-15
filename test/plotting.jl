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
        xs = vcat([0.0, 0.0, 1.0], fill(rand(), 97))
        ys = randn(100)
        result = binnedmeanandstd(xs, ys)
        @test length(result.subplots[1].series_list[2].plotattributes[:x]) == 2 # 2 bins only
    end
end

@testset "Convergence" begin
    n = 1000
    # Set the global seeds, so we don't accidentally produce a random error because of
    # how the data was randomly sampled
    Random.seed!(1234)

    target = (;
        y1 = randn(n),
        y2 = randn(n)
    )
    features = (;
        x1y1 = target.y1,
        x2y1 = target.y1 .+ 0.05randn(n),
        x3y1 = target.y1 .+ 0.1randn(n),
        x4y1 = target.y1 .+ 0.2randn(n),
        siny1 = sin.(target.y1),
        cosy1 = cos.(target.y1),
        cos2y1 = cos.(target.y1),
        xrand1 = randn(n),
        x1y2 = target.y2,
        x2y2 = target.y2 .+ 0.05randn(n),
        x3y2 = target.y2 .+ 0.1randn(n),
        x4y2 = target.y2 .+ 0.2randn(n),
        siny2 = sin.(target.y2),
        cosy2 = cos.(target.y2),
        cos2y2 = cos.(target.y2),
        xrand2 = randn(n)
    )

    labels = ["noise", "trig", "rand"]
    kwargs = (; n = 10, sz = collect(10:10:1000), group_names=labels)
    @testset "Grouped Tables" begin
        groups = [
            NamedTuple{(:x1y1, :x2y1, :x3y1)}(features),
            NamedTuple{(:siny1, :cosy1, :cos2y1)}(features),
            NamedTuple{(:xrand1, :xrand2)}(features),
        ]

        result = convergence(MutualInformation(), groups, target.y1; kwargs...)
        @test length(result.series_list) == 3

        @testset "$(labels[i])" for i in 1:3
            @test result.series_list[i].plotattributes[:label] == labels[i]
            y = result.series_list[i].plotattributes[:y]
            @test isnan(last(y))
            @test count(isnan, y) == length(groups[i])

            @test length(y) == length(groups[i]) * length(kwargs.sz) + length(groups[i])
            # Test that we're generally converging?
            # Maybe there's a better check?
            if labels[i] != "rand"
                deltas = abs.(diff(y[1:length(kwargs.sz)]))
                @test first(deltas) > last(deltas)
            end
        end
    end

    @testset "Single Matrix" begin
        groups = hcat(features.x1y1, features.siny1, features.xrand1)
        result = convergence(MutualInformation(), groups, target.y1; kwargs...)
        @test length(result.series_list) == 3

        @testset "$(labels[i])" for i in 1:3
            @test result.series_list[i].plotattributes[:label] == labels[i]
            y = result.series_list[i].plotattributes[:y]
            @test isnan(last(y))
            @test count(isnan, y) == 1

            @test length(y) == length(kwargs.sz) + 1
            # Test that we're generally converging?
            # Maybe there's a better check?
            if labels[i] != "rand"
                deltas = abs.(diff(y[1:length(kwargs.sz)]))
                @test first(deltas) > last(deltas)
            end
        end
    end

    @testset "Singular Table" begin
        groups = NamedTuple{(:x1y1, :siny1, :xrand1)}(features)
        result = convergence(MutualInformation(), groups, target.y1; kwargs...)
        @test length(result.series_list) == 3   # These should still be separate groups

        @testset "$(labels[i])" for i in 1:3
            @test result.series_list[i].plotattributes[:label] == labels[i]
            y = result.series_list[i].plotattributes[:y]
            @test isnan(last(y))
            @test count(isnan, y) == 1

            @test length(y) == length(kwargs.sz) + 1
            # Test that we're generally converging?
            # Maybe there's a better check?
            if labels[i] != "rand"
                deltas = abs.(diff(y[1:length(kwargs.sz)]))
                @test first(deltas) > last(deltas)
            end
        end
    end

    @testset "Contiguous, multiple targets" begin
        groups = NamedTuple{(:x1y1, :siny1, :xrand1)}(features)
        result = convergence(MutualInformation(), groups, target; kwargs..., contiguous = true)
        @test length(result.series_list) == 3   # These should still be separate groups

        @testset "$(labels[i])" for i in 1:3
            @test result.series_list[i].plotattributes[:label] == labels[i]
            y = result.series_list[i].plotattributes[:y]
            @test isnan(last(y))
            @test count(isnan, y) == 2

            @test length(y) == length(kwargs.sz) * 2 + 2
            # Test that we're generally converging?
            # Maybe there's a better check?
            if labels[i] != "rand"
                deltas = abs.(diff(y[1:length(kwargs.sz)]))
                @test first(deltas) > last(deltas)
            end
        end
    end

    @testset "Groups mismatch" begin
        groups = NamedTuple{(:x1y1, :siny1)}(features)
        @test_throws ArgumentError convergence(
            MutualInformation(), groups, target.y1; group_names=labels
        )
    end
end
