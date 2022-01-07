These recommendations advise on how to use this package to search for promising features to predict some target.

- Use Mutual Information (MI) as the default criterion.
- Transform the features and the targets using `FeatureRelevance.log_transform` and compute MI on the transformed values. Mathematically, the MI is unchanged under one-to-one maps, but it becomes easier to estimate [[1]](https://arxiv.org/pdf/cond-mat/0305641.pdf).
- Compare the `MI(feature, target)` to `MI(shuffle(feature), target)` to get a sense of what the MI value means, compared to noise with the same distribution (shuffled data). The `RatioToShuffled` criterion is a convenient way to do this in one step.
- Compare the `MI(feature, target)` to `ConditionalMI(feature, target, existing_feature)` to see whether the feature contains information beyond what is already present in an `existing_feature`. `CMI` is numerically unstable and the results should be taken _cum grano salis_. Compute also `MI(existing_feature, target)` and `MI(feature, existing_feature)` to help you make a decision.
- Compare the `MI(feature, target)` to the `MI(lagged_target, target)` to see how our proposed `feature` compares to lagged values of the `target`. Roughly equivalent to `MI(circshift(feature, N)[N+1:end], target[N+1:end])` The `RatioToLagged` criterion is another convenient way to do this in one step.
