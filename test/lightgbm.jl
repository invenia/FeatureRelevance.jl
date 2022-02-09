using LightGBM

using FeatureRelevance:
    PredictivePowerScore,
    GradientBoostedImportance,
    GainImportance,
    SplitImportance,
    ShapleyValues

@testset "lightgbm.jl" begin
    @testset "PredictivePowerScore" begin
        # Simple non-linear example used in
        # https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598
        X = rand(-2.0:0.1:2.0, 1000)
        y = [x^2 + rand(-0.5:0.1:0.5) for x in X]

        pps = PredictivePowerScore()
        pearson = PearsonCorrelation()

        # Pearson correlation should be close to zero
        @test abs(pearson(X, y)) < 0.1

        # PPS score for X predicting y should be over 0.5
        @test pps(X, y) > 0.5
        # But the score for y predicting X should be close to zero, since the mean of the
        # conditional distribution `P(y|X)` (this is essentially what is predicted by the
        # underlying regressor) is invariant in `X`
        @test pps(y, X) < 0.1
    end

    @testset "GradientBoostedImportance" begin
        X = rand(-5.0:0.1:5.0, 100000)
        df = DataFrame(
            :x1 => X + rand(0.0:0.1:0.5, length(X)),
            :x2 => X + rand(-5.0:0.1:5.0, length(X)),
            :target => X.^2,
        )

        prev = zeros(3)
        for alg in (GainImportance(), SplitImportance())
            idx, scores = selection(alg, df[:, [:x1, :x2]], df[:, [:target]])
            # Since our features are progressively more noisy we can just check that
            # that the scores are in sorted in descending order
            @test issorted(round.(scores; digits=10); rev=true)

            # Test that our scores aren't identical between importance types
            @test scores != prev
        end

        # Test that this also works with `report`
        alg = GradientBoostedImportance(:gain)
        r = DataFrame(report(alg, df[:, [:x1, :x2]], df[:, [:target]]))
        f, scores = r[:, :feature], r[:, :score]
        @test f == [:x1, :x2]
        @test scores[1] > scores[2]

        # Test with array inputs
        r2 = report(alg, Tables.matrix(df[:, [:x1, :x2]]), df.target)
        @test DataFrame(r2).score == r.score

        # Test dropping redundant features
        df.x11 = df.x1 # identical df.x1, should be dropped
        alg = GradientBoostedImportance(; importance_type=:gain, iterations=0, positive=true)
        r = DataFrame(report(alg, df[:, [:x1, :x11, :x2]], df[:, [:target]]))
        f, scores = r[:, :feature], r[:, :score]
        @test issetequal(f, [:x1, :x2])
    end

    @testset "ShapleyValues" begin
        X = rand(-5.0:0.1:5.0, 100000)
        df = DataFrame(
            :x1 => -X,
            :x2 => X + rand(0.0:0.1:0.5, length(X)),
            :x3 => X + rand(-5.0:0.1:5.0, length(X)),
            :x4 => zeros(length(X)),
            :target => X.^2,
        )

        idx, scores = selection(
            ShapleyValues(),
            df[:, [:x1, :x2, :x3, :x4]],
            df[:, [:target]]
        )

        # Since our features are progressively more noisy we can just check that
        # that the magnitude of the scores are sorted in descending order
        @test issorted(abs.(round.(scores; digits=10)); rev=true)
        @test length(idx) == 4
        @test iszero(last(scores))

        # Test dropping completely useless feature
        idx, scores = selection(
            ShapleyValues(; positive=true),
            df[:, [:x1, :x2, :x3, :x4]],
            df[:, [:target]]
        )
        @test length(idx) == 3
        @test all(!iszero, scores)

        # Test that this also works with `report`
        alg = ShapleyValues()
        r = DataFrame(report(alg, df[:, [:x1, :x2, :x3]], df[:, [:target]]))
        f, scores = r[:, :feature], r[:, :score]
        @test f == [:x1, :x2, :x3]
        @test abs(scores[1]) > abs(scores[3])

        # Test with array inputs
        r2 = report(alg, Tables.matrix(df[:, [:x1, :x2, :x3]]), df.target)
        @test DataFrame(r2).score == r.score
    end

    @testset "_to_real_array" begin
        X = hcat(
            rand(-2.0:0.1:2.0, 10000),
            rand(-0.5:0.1:0.5, 10000),
            rand(5.0:0.5:100.0, 10000),
        )

        df = DataFrame(
            :x1 => X[:, 1],
            :x2 => X[:, 2],
            :x3 => X[:, 3],
        )
        sa = view(X, 1:10000, 1:3)
        sa2 = view(X, 1:10000, 2)
        ka = wrapdims(X; r=1:10000, c=[:x1, :x2, :x2])

        @test FeatureRelevance._to_real_array(df) == X          # Table
        @test FeatureRelevance._to_real_array(sa) == X          # SubArray
        @test FeatureRelevance._to_real_array(sa2) == reshape(X[1:10000, 2], 10000, 1)
        @test FeatureRelevance._to_real_array(ka) == X          # KeyedArray
        @test FeatureRelevance._to_real_array(eachcol(X)) == X  # Generator
    end
end
