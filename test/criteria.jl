@testset "criteria.jl" begin

    x = rand(100)
    y1 = x + 0.1*rand(100)
    y2 = x + 0.2*rand(100)

    for criterion in (
        MutualInformation(),
        NormalisedMutualInformation(),
        PearsonCorrelation(),
        SpearmanCorrelation(),
    )
        @test criterion(x, y1) > criterion(x, y2)
    end

    criterion = ConditionalMutualInformation()
    z = y1
    @test criterion(x, y2, z) > criterion(x, y1, z)
end
