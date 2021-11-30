using LightGBM

using FeatureRelevance: PredictivePowerScore, GradientBoostedImportance, GainImportance, SplitImportance

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
        X = rand(-2.0:0.1:2.0, 10000)
        df = DataFrame(
            :x1 => X,
            :x2 => [x + rand(-0.5:0.1:0.5) for x in X],
            :x3 => [x + rand(-1.0:0.1:1.0) for x in X],
            :target => [x^2 for x in X],
        )

        prev = zeros(3)
        for alg in (GainImportance(), SplitImportance())
            idx, scores = selection(alg, df[:, [:target]], df[:, [:x1, :x2, :x3]])
            # Since our features are progressively more noisy we can just check that
            # that the scores are in sorted in descending order
            @test issorted(scores; rev=true)

            # Test that our scores aren't identical between importance types
            @test scores != prev
        end

        # Test that this also works with `report`
        alg = GradientBoostedImportance(:gain)
        r = DataFrame(report(alg, df[:, [:target]], df[:, [:x2, :x3]]))
        f, scores = r[:, :feature], r[:, :score]
        @test f == [:x2, :x3]
        @test scores[1] > scores[2]

        @test_throws TaskFailedException report(
            alg, df.target, Tables.matrix(df[:, [:x1, :x2]])
        )
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
        sa2 = view(X, 1:10000, 1:2)
        ka = wrapdims(X; r=1:10000, c=[:x1, :x2, :x2])

        @test FeatureRelevance._to_real_array(df) == X          # Table
        @test FeatureRelevance._to_real_array(sa) == X          # SubArray
        @test FeatureRelevance._to_real_array(ka) == X          # KeyedArray
        @test_throws MethodError FeatureRelevance._to_real_array(eachcol(X))  # Generator
    end
end
