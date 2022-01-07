# FAQ

## Why is the API so loosely typed?

Our API is intentionally designed with loose type constraints on functions by default.
Many methods for `evaluate` and `selection` simply need to know that `target` is an iterable with a length.
Similarly, `features` just needs to be an iterable (e.g., `Vector`, `NamedTuple`, `Generator`) of `feature` iterables.
This design also makes it easier to support more complex [`table`](https://tables.juliadata.org/stable/) and
[`array`](https://github.com/mcabbott/AxisKeys.jl) types without depending on specific packages/implementations.


## What if I require more specific type?

While it's recommended that you avoid overly restrictive type constraints, sometimes your code requires this.
In these cases, we recommended including methods which perform generic conversions to your required type.
An example of this can be seen in `src/lightgbm.jl`, where the underlying library is restricted to
operating on dense arrays of reals.


## LightGBM.jl extensions?

[LightGBM.jl](https://github.com/IQVIA-ML/LightGBM.jl) extensions are provided via [Requires.jl](https://github.com/JuliaPackaging/Requires.jl).
As such, you'll need to have LightGBM installed and loaded prior to loading FeatureRelevance.

```julia
using Pkg; Pkg.add("LightGBM")
```

**WARNING** On `macos` you'll also need to manually install the `libomp` binary
dependency in your shell.

```sh
brew install libomp
```

In the future, BinaryBuilder and BinaryProvider should be able to handle this for us [[1]](https://github.com/IQVIA-ML/LightGBM.jl/issues/112).

```@setup lightgbm
n = 10000
target = randn(n)
features = hcat(
    target,
    target .+ 0.05randn(n),
    target .+ 0.1randn(n),
    target .+ 0.2randn(n),
    sin.(target),
    cos.(target),
    cos.(target),
    randn(n),
)
```

```@example lightgbm
using LightGBM
using FeatureRelevance: report, Top, PredictivePowerScore

feat_imp = report(PredictivePowerScore(), Top(5), target, features)
```

We can also use the split and gain importance algorithms.
```@example lightgbm
using FeatureRelevance: SplitImportance
report(SplitImportance(), target, features)
```
