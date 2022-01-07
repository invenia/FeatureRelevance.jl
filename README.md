# FeatureRelevance.jl

[![Docs: Latest](https://img.shields.io/badge/docs-latest-blue.svg)](http://docs.invenia.ca/invenia/research/FeatureRelevance.jl)

A package for scoring and selecting relevant features for 1 or more target variables.

## Quickstart

Tip: Start `julia` with `--threads N` to utilise multithreading functionality.

```@setup quickstart
n = 10000
y = (y1=randn(n), y2=rand(n))
X = (;
    x1 = y.y1 .+ 0.05randn(n),
    x2 = y.y2 .+ 0.1randn(n),
    xsin = sin.(y.y1),
    xcos = cos.(y.y2),
    xrand = randn(n),
)
```

Load `report` function and `GreedyMRMR` selection algorithm.
We'll also load DataFrames.jl to improve readability.
```@repl quickstart
using FeatureRelevance: report, GreedyMRMR
using DataFrames
```

Wrap some existing features and targets in DataFrames.
```@repl quickstart
targets = DataFrame(y)
features = DataFrame(X)
```

Finally, generate a report of the top 5 non-redundant features for each target variable.
```@repl quickstart
report(GreedyMRMR(5; positive=true), targets, features) |> DataFrame
```
