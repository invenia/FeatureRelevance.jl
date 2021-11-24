@testset "criteria.jl" begin

    x = rand(10000)
    y1 = x + 0.1*rand(10000)
    y2 = x + 0.2*rand(10000)

    for criterion in (
        MutualInformation(),
        NormalisedMutualInformation(),
        PearsonCorrelation(),
        SpearmanCorrelation(),
    )
        @test criterion(x, y1) > criterion(x, y2)
        @test evaluate(criterion, x, y1) == criterion(x, y1)
    end

    @testset "RatioToShuffled()" begin
        criterion = RatioToShuffled(MutualInformation())
        @test criterion(x, y1) > criterion(x, y2)

        @test 0.9 < criterion(x, rand(size(x)...)) < 1.1
    end

    @testset "ConditionalMutualInformation()" begin
        criterion = ConditionalMutualInformation()
        z = y1
        @test criterion(x, y2, z) > criterion(x, y1, z)
        @test evaluate(criterion, x, y1, z) == criterion(x, y1, z)
    end
end
