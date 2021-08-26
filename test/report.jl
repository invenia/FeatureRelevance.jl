@testset "report.jl" begin
    @testset "single target" begin
        targets = DataFrame(:t1 => rand(4))
        features = DataFrame(rand(4, 10), :auto)
        m = DataFrame(report(PearsonCorrelation(), Top(2), targets, features))
        @test size(m) == (2, 3) # (nselect, 3)
        @test propertynames(m) == [:target, :feature, :score]
        @test count(r -> r === :t1, m.target) == 2
    end

    @testset "multiple targets" begin
        targets = DataFrame(:t1 => rand(4), :t2 => rand(4))
        features = DataFrame(rand(4, 10), :auto)
        m = DataFrame(report(PearsonCorrelation(), Top(2), targets, features))
        @test size(m) == (4, 3) # (ntargets x nselect, 3)
        @test propertynames(m) == [:target, :feature, :score]
        @test issetequal(m.target, propertynames(targets))
        @test count(r -> r === :t1, m.target) == 2
    end
end
