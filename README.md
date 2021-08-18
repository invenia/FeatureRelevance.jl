# FeatureRelevance.jl

Example usage for different metrics/selection approaches:

```
using FeatureRelevance
using DataFrames

n = 1000
target_df = DataFrame(z = rand(n))

source_df = DataFrame(a = target_df.z .+ 0.1randn(n))
source_df[!,:b] = source_df.a .+ 0.1randn(n)
source_df[!,:c] = source_df.b .+ 0.1randn(n)
source_df[!,:d] = source_df.c .+ 0.1randn(n)
source_df[!,:e] = source_df.d .+ 0.1randn(n)

n_select = 5

feat_imp = FeatureRelevance.select(Top(n_select,MutualInformation()), target_df, source_df)
feat_imp = FeatureRelevance.select(Top(n_select,NormalisedMutualInformation()), target_df, source_df)
feat_imp = FeatureRelevance.select(Top(n_select,PearsonCorrelation()), target_df, source_df)
feat_imp = FeatureRelevance.select(Top(n_select,SpearmanCorrelation()), target_df, source_df)

feat_imp = FeatureRelevance.select(GreedyMRMR(n_select), target_df, source_df)
feat_imp = FeatureRelevance.select(GreedyJMI(n_select), target_df, source_df)

feat_imp = FeatureRelevance.select(Top(n_select,PredictivePowerScore()), target_df, source_df)
feat_imp = FeatureRelevance.select(ShapleyValues(), target_df, source_df)
feat_imp = FeatureRelevance.select(GainImportance(), target_df, source_df)
feat_imp = FeatureRelevance.select(SplitImportance(), target_df, source_df)
```
