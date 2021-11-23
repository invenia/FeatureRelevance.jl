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

## Recommendations

These recommendations advise on how to use this package to search for promising features to predict some target.

1) Use Mutual Information (MI) as the default criterion.

2) Transform the features and the targets using `FeatureRelevance.log_transform` and compute MI on the transformed values.
Mathematically, the MI is unchanged under one-to-one maps, but it becomes easier to estimate [1](https://arxiv.org/pdf/cond-mat/0305641.pdf).

3) Compare the `MI(feature, target)` to `MI(shuffle(feature), target)` to get a sense of what the MI numbers mean.

4) Compare the `MI(feature, target)` to `ConditionalMI(feature, target, existing_feature)` to see whether the feature contains information beyond what is already present in `existing_feature`.
Note, however, that `CMI` is numerically unstable and the results should be taken _cum grano salis_.
Compute also `MI(existing_feature, target)` and `MI(feature, existing_feature)` to help you make a decision.

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
using FeatureRelevance: PredictivePowerScore, GainImportance, SplitImportance

feat_imp = FeatureRelevance.report(PredictivePowerScore(), Top(n_select), target_df, source_df)
feat_imp = FeatureRelevance.report(GainImportance(), target_df, source_df)
feat_imp = FeatureRelevance.report(SplitImportance(), target_df, source_df)
```


## Future (to be added)

```julia
feat_imp = FeatureRelevance.report(ShapleyValues(), target_df, source_df)
```
