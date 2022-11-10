@testset "report.jl" begin
    # Really simple alg for testing basic report input/output behaviour
    ntop = Top(; criterion=PearsonCorrelation(), n=2)
    alltop = Top(; criterion=PearsonCorrelation())

    @testset "single target" begin
        targets = DataFrame(:t1 => rand(4))
        features = DataFrame(rand(4, 10), :auto)
        m = DataFrame(report(ntop, features, targets))

        @test size(m) == (2, 4) # (nselect, 4)
        @test propertynames(m) == [:target, :feature, :n_obs, :score]
        @test count(r -> r === :t1, m.target) == 2

        m = DataFrame(report(alltop, features, targets))
        @test size(m) == (10, 4) # (all, 4)
        @test propertynames(m) == [:target, :feature, :n_obs, :score]
        @test count(r -> r === :t1, m.target) == 10
    end

    @testset "multiple targets" begin
        targets = DataFrame(:t1 => rand(4), :t2 => rand(4))
        features = DataFrame(rand(4, 10), :auto)
        m = DataFrame(report(ntop, features, targets))

        @test size(m) == (4, 4) # (ntargets x nselect, 4)
        @test propertynames(m) == [:target, :feature, :n_obs, :score]
        @test issetequal(m.target, propertynames(targets))
        @test count(r -> r === :t1, m.target) == 2

        m = DataFrame(report(alltop, features, targets))
        @test size(m) == (20, 4) # (ntargets x all, 4)
        @test propertynames(m) == [:target, :feature, :n_obs, :score]
        @test issetequal(m.target, propertynames(targets))
        @test count(r -> r === :t1, m.target) == 10
    end

    @testset "$matvec inputs" for (matvec, data) in (
            ("matrix", rand(100, 1)),
            ("vector", rand(100)),
        )
        targets = rand(100, 1)
        features = rand(100, 8)
        features[:, 8] = targets + 0.4*rand(100)
        features[:, 5] = targets + 0.9*rand(100)
        m = DataFrame(report(ntop, features, targets))

        @test propertynames(m) == [:target, :feature, :n_obs, :score]
        @test m[:, :feature] == [8, 5]
    end
end
