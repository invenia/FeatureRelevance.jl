@testset "relevance.jl" begin
    criterion = PearsonCorrelation()

    @testset "all missing" begin
        @test (0, missing) === relevance(criterion, [missing, missing], [1.0, 2.0])
        @test (0, missing) === relevance(criterion, [missing, 1.0], [1.0, missing])
    end

    @testset "some missing" begin
        # This returns the number of elements compared and the relevance score
        @test (2, 1.0) === relevance(criterion, [missing, 1.0, 2.0], [1.0, 2.0, 3.0])
    end

    @testset "repeated values" begin
        @test (0, missing) === relevance(criterion, [missing, 1.0, 1.0], [1.0, 2.0, 3.0])
    end
end
