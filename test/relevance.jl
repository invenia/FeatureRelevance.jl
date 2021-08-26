@testset "relevance.jl" begin
    criterion = PearsonCorrelation()

    @testset "all missing" begin
        @test missing === relevance(criterion, [missing, missing], [1.0, 2.0])
        @test missing === relevance(criterion, [missing, 1.0], [1.0, missing])
    end

    @testset "some missing" begin
        @test 1.0 === relevance(criterion, [missing, 1.0, 2.0], [1.0, 2.0, 3.0])
    end

    @testset "repeated values" begin
        @test missing === relevance(criterion, [missing, 1.0, 1.0], [1.0, 2.0, 3.0])
    end
end
