# Usage

In general, the FeatureRelevance.jl API can be broken down into:

- Evaluating the relevance between a specific feature and target (1:1) [[1]](@ref evaluating)
- Selecting and reporting the most relevant features for one or more targets (m:n) [[2]](@ref selecting)


## [Evaluating the Relevance](@id evaluating)

The first step in any feature selection analysis is to identify an appropriate measure of relevance.
Similar to [Distances.jl](https://github.com/JuliaStats/Distances.jl) and [Metrics.jl](https://gitlab.invenia.ca/invenia/Metrics.jl),
FeatureRelevance.jl uses a `Criterion` type to encapsulate the metric settings.
Each subtype defines an `evaluate(criterion, x, y)` method to generate a relevance score between equal-length iterators `x` and `y`.
A generic `relevance(criterion, x, y, z...)` wrapper function is used to:

1. Drop missing observations
2. Check that each iterator has more than 1 unique value

Depending on the kinds of relationships you wish to detect and the properties of your downstream model
(e.g., linear regression, NN, GP), your `Criterion` metric can be vital in finding useful features.

For example, if we generate some test features as a linear, trig, log and sqrt permutation over a target,
we can see that the relative scores are quite different between pearson correlation and mutual information.

```@example usage-relevance
using Plots
using LightGBM
using FeatureRelevance:
    relevance,
    PearsonCorrelation,
    PredictivePowerScore,
    NormalisedMutualInformation

pcor = PearsonCorrelation()
pps = PredictivePowerScore()
mi = NormalisedMutualInformation()
n = 10000
y = randn(n)
xlin = y .+ 0.2randn(n)
xtrig = sin.(y)
xlog = log.(abs.(y))
xsqrt = sqrt.(abs.(y))

iter = Iterators.product([pcor, pps, mi], [xlin, xtrig, xlog, xsqrt])
results = Iterators.map(iter) do (criterion, x)
    relevance(criterion, x, y)
end |> collect

heatmap(["xlin", "xtrig", "xlog", "xsqrt"], ["pcor", "pps", "mi"], results)
```

## [Selecting and Reporting the Most Relevant Features](@id selecting)

Out next step is to filter out our highest value features for a given target
An `Algorithm` describes the process by which we select the most relevant features.
In the same way that criteria must implement the `evaluate` method, algorithms implement `selection` methods.
Similarly, just like the `relevance` wrapper, FeatureRelevance.jl provides a convenient `report` wrapper which:

1. Handles running `selection` for multiple targets
2. Handles various input formats such as tables and matrices

For simplicity we'll generating some fake data with very clear properties we might care about.

1. Varying degrees of noise added to the desired target variable
2. Some features have a linear relationship to the target while others are non-linear.
3. One variable is completely unrelated to our target.
4. Another variable is just a duplicate.

```@example usage-selection
using DataFrames
using LightGBM
using FeatureRelevance: report, Top, Greedy, SplitImportance

n = 10000
target = (; p = randn(n))
features = (;
    xp = target.p,
    xp1 = target.p .+ 0.05randn(n),
    xp2 = target.p .+ 0.1randn(n),
    xp3 = target.p .+ 0.2randn(n),
    xpsin = sin.(target.p),
    xpcos = cos.(target.p),
    xpcos2 = cos.(target.p),
    xrand = randn(n),
)
```

The most basic algorithm is simply to select the top features independent of one another.
```@example usage-selection
report(Top(5), target, features) |> DataFrame
```
However, it could be the case that we do not want to do any filtering and just return all values. 

`report(ALL, target, features) |> DataFrame`

In practice, we usually want to filter, as there are `O(T*F)` (number of targets, number of features) which can be large.  

NOTE: `report` returns a [`Tables.columntable`](https://tables.juliadata.org/stable/#Tables.columntable), so you can always convert the result to whatever table type you require.

However, we may be concerned about redundant information between features, so the `Greedy`
algorithm may be desirable at the cost of increased execution time.

```@example usage-selection
report(Greedy(; n=5, β=true, γ=true, positive=true), target, features) |> DataFrame
```
We can see that the several scores are lower due to redundancy between the features and the
second copy of `cos.(target)` has been completely excluded.

Alternatively, we may want to consider an algorithm which relies on a fitted estimator to select relevant features.
```@example usage-selection
report(SplitImportance(; iterations=10), target, features) |> DataFrame
```
This successfully assigns the strongest signal to `target`, `sin.(target)` and `cos.(target)` and ignores redundant information.
However, this requires that you have LightGBM.jl and its dependencies installed already.
See the FAQ on setting up LightGBM.jl for more details.

Finally, what if we had multiple targets

```@example usage-selection
target = (; p = target.p, q = log.(abs.(target.p)))
report(SplitImportance(; iterations=10), target, features) |> DataFrame
```
