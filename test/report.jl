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

    @testset "$matvec inputs" for (matvec, data) in (
            ("matrix", rand(100, 1)),
            ("vector", rand(100)),
        )
        targets = rand(100, 1)
        features = rand(100, 8)
        features[:, 8] = targets + 0.4*rand(100)
        features[:, 5] = targets + 0.9*rand(100)
        m = DataFrame(report(PearsonCorrelation(), Top(2), targets, features))

        @test propertynames(m) == [:target, :feature, :score]
        @test m[:, :feature] == [8, 5]
    end
end
