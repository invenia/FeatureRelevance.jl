rng = MersenneTwister(1)

@testset "Feature selection" begin

    # Test behaviour of methods
    # Repeat to ensure robustness regardless of rng results
    num_repeats = 10
    for i in 1:num_repeats
        N = 100
        df = DataFrame(
             :target => randn(rng, N),
        )
        df[!,:x1] = df[:, :target] .+ 0.1randn(rng, N)  # Most informative
        df[!,:x2] = df[:, :x1] # Duplicate
        df[!,:x3] = df[:, :target] + 0.2randn(rng, N)  # Second most informative

        targets = df[:,[1]]
        features = df[:,2:end]

        methods = (
            Top(2, NormalisedMutualInformation()),
            GreedyMRMR(2),
            GreedyJMI(2),
       )
        for m in methods
            @testset "$m basic properties" begin
                @testset "Safety checks" begin
                    # Pass in target as a feature
                    @test_throws ArgumentError _select_single_target(m, df, df)
                    # Try to use more than one target
                    @test_throws ArgumentError _select_single_target(m, df[:,1:2], df[:,3:end])
                    # Try to pass in bare array as target
                    @test_throws MethodError _select_single_target(m, df[:,1], df[:,2:end])
                end

                selected_features = _select_single_target(m, targets, features)[:,:feature]
                @testset "Find most relevant" begin
                    @test :x1 ∈ selected_features
                end
            end
        end

        @testset "Top N" begin
            # Top N will just find the top features, even though they are duplicates
            f = sort(
                _select_single_target(Top(2, MutualInformation()), targets, features)[:,:feature]
            )
            @test f == [:x1, :x2]
        end

        # Both MRMR and JMI should be clever enough to get x1/2 and x3
        # They should NOT pick both x1 and x2 since this is a straight duplicate
        for m in (GreedyMRMR(2), GreedyJMI(2))
            @testset "$m correct" begin
                f = sort(_select_single_target(m, targets, features)[:,:feature])
                @test f != [:x1, :x2]
                @test (f == [:x1, :x3]) | (f == [:x2, :x3])
            end
        end
    end

    # Test calculated scores
    @testset "Scores single target" begin
        N = 10000 # N=100 results in occasional failures due to randomness
        df = DataFrame(
            :target => randn(rng, N),
        )
        #= These values are set to test that MRMR and JMI will return a
        different order of features than the MI approach by prioritizing
        features that don't contain redundant information =#
        # x1 is the target with noise
        df[!,:x1] = df[:, :target] + 0.1randn(rng, N)
        # x2 is x1 with some noise
        df[!,:x2] = df[:, :x1] + 0.1randn(rng, N)
        # x3 = target - x1, results in a feature without redundant information
        df[!,:x3] = df[:, :target] .- df[!,:x1]

        targets = df[:,[1]]
        features = df[:,2:end]
        # Calculate all the pairwise MI
        mi = Dict()
        cmi = Dict()
        for a in Symbol.(names(df))
            mi[a]= Dict()
            cmi[a]= Dict()
            for b in Symbol.(names(df))
                mi[a][b] = MutualInformation()(df[:,a], df[:,b])
                cmi[a][b] = ConditionalMutualInformation()(
                    df[:,a], df[:,b];
                    conditioned_variable=df[:,1]
                )
            end
        end

        m = _select_single_target(Top(3, MutualInformation()), targets, features)
        f, scores = m[:,:feature], m[:,:score]
        @test f == [:x1, :x2, :x3]
        @test scores[1] ≈ mi[:target][:x1]
        @test scores[2] ≈ mi[:target][:x2]
        @test scores[3] ≈ mi[:target][:x3]

        #= NB with MRMR later scores can be higher than the previous scores
        and this can cause the tests to fail when we assume a specific order.
        One possible reason for this is that the redundancy is calculated as
        the average of the mutual information with the previously selected
        features and is subtracted from the multual information of the candiate
        feature with the target feature =#

        m = _select_single_target(GreedyMRMR(3), targets, features)
        f, scores = m[:,:feature], m[:,:score]
        @test f == [:x1, :x3, :x2]
        @test scores[1] ≈ mi[:target][:x1]
        @test scores[2] ≈ mi[:target][:x3] - mi[:x1][:x3]
        @test scores[3] ≈ mi[:target][:x2] - 1.0/2.0 * (mi[:x1][:x2] + mi[:x3][:x2])

        # randomly fails
        m = _select_single_target(GreedyJMI(3), targets, features)
        f, scores = m[:,:feature], m[:,:score]
        @test f == [:x1, :x3, :x2]
        @test scores[1] ≈ mi[:target][:x1]
        @test scores[2] ≈ mi[:target][:x3] -
            mi[:x1][:x3] +
            cmi[:x1][:x3]
        @test scores[3] ≈ mi[:target][:x2] -
            1.0/2.0 * (mi[:x1][:x2] + mi[:x3][:x2]) +
            1.0/2.0 * (cmi[:x1][:x2] + cmi[:x3][:x2])
    end

    @testset "Select multiple" begin
        targets = DataFrame(:t1 => rand(4), :t2 => rand(4))
        features = DataFrame(rand(4, 10), :auto)
        m = select(Top(3, PearsonCorrelation()), targets, features)
        @test m isa Dict{Symbol, DataFrame}
        @test sort(collect(keys(m))) == Symbol.(sort(names(targets)))
        @test Symbol.(names(m[:t1])) == [:feature, :score]
        @test nrow(m[:t1]) == 3
    end

    # @testset "Snapshot feature selection" begin
    #     target_nodes = Symbol.([
    #         "ALTE.ALTE",
    #         "ALTW.BART1NIPS",
    #         "AMMO.AUDRN77",
    #     ])
    #
    #     for fsm in (GreedyMRMR(2), GreedyJMI(2))
    #         @testset "Method = $fsm" begin
    #             mapping = get_snapshot_features(
    #                 target_nodes,
    #                 fsm;
    #             )
    #             @test mapping isa Dict{Symbol, Vector{Symbol}}
    #             @test sort(collect(keys(mapping))) == target_nodes
    #             @test Symbol("price_delta_ALTE.ALTE_lag120") ∈ mapping[Symbol("ALTE.ALTE")]
    #         end
    #     end
    # end
end
