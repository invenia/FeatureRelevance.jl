@testset "preprocess.jl" begin
    @test log_transform(0) == 0.0

    v = randn()
    @test log_transform(v) == -log_transform(-v)
    @test log_transform(ℯ^v) > v
    @test log_transform(-ℯ^v) < -v

    @test log_transform([0.0, v]) == [0.0, log_transform(v)]
end
