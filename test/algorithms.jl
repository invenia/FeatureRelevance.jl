@testset "algorithms.jl" begin
    @testset "duplicate features" begin
        # Test behaviour of algorithms
        # Repeat to ensure robustness regardless of rng results
        num_repeats = 10
        for i in 1:num_repeats
            N = 100
            df = DataFrame(:target => randn(rng, N))
            df[!, :x1] = df[:, :target] .+ 0.1randn(rng, N)  # Most informative
            df[!, :x2] = df[:, :x1] # Duplicate
            df[!, :x3] = df[:, :target] + 0.2randn(rng, N)  # Second most informative

            targets = df[:, [1]]
            features = df[:, 2:end]

            @testset "Top N" begin
                # Top N will just find the top features, even though they are duplicates
                r = DataFrame(report(MutualInformation(), Top(2), targets, features))
                @test r.feature == [:x1, :x2]
            end

            # Both MRMR and JMI should be clever enough to get x1/2 and x3
            # They should NOT pick both x1 and x2 since this is a straight duplicate
            for m in (GreedyMRMR(2), GreedyJMI(2))
                @testset "$m correct" begin
                    r = DataFrame(report(m, targets, features))
                    @test r.feature != [:x1, :x2]
                    @test (r.feature == [:x1, :x3]) || (r.feature == [:x2, :x3])
                end
            end
        end
    end

    @testset "Value of scores" begin
        N = 10000 # N=100 results in occasional failures due to randomness
        df = DataFrame(:target => randn(rng, N))
        #= These values are set to test that MRMR and JMI will return a
        different order of features than the MI approach by prioritizing
        features that don't contain redundant information =#
        # x1 is the target with noise
        df[!, :x1] = df[:, :target] + 0.1randn(rng, N)
        # x2 is x1 with some noise
        df[!, :x2] = df[:, :x1] + 0.1randn(rng, N)
        # x3 = target - x1, results in a feature without redundant information
        df[!, :x3] = df[:, :target] .- df[!, :x1]

        targets = df[:, [1]]
        features = df[:, 2:end]
        # Calculate all the pairwise MI
        mi = Dict()
        cmi = Dict()
        for a in Symbol.(names(df))
            mi[a] = Dict()
            cmi[a] = Dict()
            for b in Symbol.(names(df))
                mi[a][b] = MutualInformation()(df[:, a], df[:, b])
                cmi[a][b] = ConditionalMutualInformation()(df[:, a], df[:, b], df[:, 1])
            end
        end

        @testset "MutualInformation, Top(3)" begin
            m = DataFrame(report(MutualInformation(), Top(3), targets, features))
            f, scores = m[:, :feature], m[:, :score]
            @test f == [:x1, :x2, :x3]
            @test scores[1] ≈ mi[:target][:x1]
            @test scores[2] ≈ mi[:target][:x2]
            @test scores[3] ≈ mi[:target][:x3]
        end

        #= NB with MRMR later scores can be higher than the previous scores
        and this can cause the tests to fail when we assume a specific order.
        One possible reason for this is that the redundancy is calculated as
        the average of the mutual information with the previously selected
        features and is subtracted from the multual information of the candiate
        feature with the target feature =#

        @testset "GreedyMRMR" begin
            m = DataFrame(report(GreedyMRMR(3), targets, features))
            f, scores = m[:, :feature], m[:, :score]
            @test f == [:x1, :x3, :x2]
            @test scores[1] ≈ mi[:target][:x1]
            @test scores[2] ≈ mi[:target][:x3] - mi[:x1][:x3]
            @test scores[3] ≈ mi[:target][:x2] - 1.0 / 2.0 * (mi[:x1][:x2] + mi[:x3][:x2])
        end

        @testset "GreedyJMI" begin
            # randomly fails
            m = DataFrame(report(GreedyJMI(3), targets, features))
            f, scores = m[:, :feature], m[:, :score]
            @test f == [:x1, :x3, :x2]
            @test scores[1] ≈ mi[:target][:x1]
            @test scores[2] ≈ mi[:target][:x3] - mi[:x1][:x3] + cmi[:x1][:x3]
            @test scores[3] ≈
                  mi[:target][:x2] - 1.0 / 2.0 * (mi[:x1][:x2] + mi[:x3][:x2]) +
                  1.0 / 2.0 * (cmi[:x1][:x2] + cmi[:x3][:x2])
        end
    end

    @testset "indexing bug (#16)" begin
        input = allowmissing(rand(12, 10))
        input[:, 1:2] .= 1.0
        output = rand(12, 2)

        idx, scores = FeatureRelevance.selection(GreedyMRMR(8), output[:, 1], eachcol(input))
        @test isdisjoint(idx, [1, 2])

        # Test missing input causing relevance to be missing
        input[:, 1] .= missing
        idx, scores = FeatureRelevance.selection(GreedyMRMR(8), output[:, 1], eachcol(input))
        @test isdisjoint(idx, [1, 2])
    end
end
