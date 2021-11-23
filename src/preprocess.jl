"""
    log_transform(x)

Log-based activation function. Used to preprocess features and targets when computing
mutual information.

log_transform(x) = sign(x) * (log(abs(x) + 1))
"""
log_transform(x) = sign(x) * (log(abs(x) + 1))
log_transform(x::AbstractArray) = log_transform.(x)
