using LightGBM

using FeatureRelevance: PredictivePowerScore

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
        # But the score for y predicting X should be similar to pearson correlation
        @test pps(y, X) < 0.1
    end
end
