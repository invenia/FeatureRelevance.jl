# FeatureRelevance.jl

[![Docs: Latest](https://img.shields.io/badge/docs-latest-blue.svg)](http://docs.invenia.ca/invenia/research/FeatureRelevance.jl)


## Usage

Using different metrics/selection approaches.
Start `julia` with `--threads N` to utilise multithreading functionality.

```julia
using FeatureRelevance: MutualInformation, NormalisedMutualInformation, PearsonCorrelation, SpearmanCorrelation
using FeatureRelevance: Top, GreedyMRMR, GreedyJMI
using DataFrames

n = 1000
target_df = DataFrame(z = rand(n))

source_df = DataFrame(a = target_df.z .+ 0.1randn(n))
source_df[!,:b] = source_df.a .+ 0.1randn(n)
source_df[!,:c] = source_df.b .+ 0.1randn(n)
source_df[!,:d] = source_df.c .+ 0.1randn(n)
source_df[!,:e] = source_df.d .+ 0.1randn(n)

n_select = 3

feat_imp = FeatureRelevance.report(MutualInformation(), Top(n_select), target_df, source_df)
feat_imp = FeatureRelevance.report(NormalisedMutualInformation(), Top(n_select), target_df, source_df)
feat_imp = FeatureRelevance.report(PearsonCorrelation(), Top(n_select), target_df, source_df)
feat_imp = FeatureRelevance.report(SpearmanCorrelation(), Top(n_select), target_df, source_df)

feat_imp = FeatureRelevance.report(GreedyMRMR(n_select), target_df, source_df)
feat_imp = FeatureRelevance.report(GreedyJMI(n_select), target_df, source_df)
```

## Experimental

Some functionality requires having LightGBM.jl installed and loaded.

```julia
using Pkg; Pkg.add("LightGBM")
```

**WARNING** On `macos` you'll also need to manually install the `libomp` binary
dependency in your shell.

```sh
brew install libomp
```

In the future, BinaryBuilder and BinaryProvider should be able to handle this for us [[1]](https://github.com/IQVIA-ML/LightGBM.jl/issues/112).

```julia
using LightGBM
using FeatureRelevance: PredictivePowerScore

feat_imp = FeatureRelevance.report(PredictivePowerScore(), Top(n_select), target_df, source_df)
```


## Future

```julia
feat_imp = FeatureRelevance.report(ShapleyValues(), target_df, source_df)
feat_imp = FeatureRelevance.report(GainImportance(), target_df, source_df)
feat_imp = FeatureRelevance.report(SplitImportance(), target_df, source_df)
```
