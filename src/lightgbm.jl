struct ShapleyValues <: RandomForestMethod end

function (::ShapleyValues)(x, y; kwargs...)
    return shapley(x, y; kwargs...)
end

function shapley(x,y; kwargs...)
    # split into 80/20 train/test
    train_idx = 1:Int64(floor(length(y)*0.8))
    test_idx = (maximum(train_idx)+1):length(y)
    train_x = x[train_idx, :]
    train_y = y[train_idx]
    test_x = x[test_idx, :]
    test_y = y[test_idx]

    estimator = LGBMRegression()
    train_report = LightGBM.fit!(estimator, train_x, train_y)
    preds = LightGBM.predict(estimator, test_x, predict_type=3)
    return mean(reshape(preds, size(train_x,2)+1,:), dims=2)[1:end-1]
end


struct GainImportance <: RandomForestMethod end

function (::GainImportance)(x, y; kwargs...)
    return gain_importance(x, y; kwargs...)
end

function gain_importance(x, y; kwargs...)
    estimator = LGBMRegression()
    LightGBM.fit!(estimator, x,y)
    return LightGBM.gain_importance(estimator)
end


struct SplitImportance <: RandomForestMethod end

function (::SplitImportance)(x, y; kwargs...)
    return split_importance(x, y; kwargs...)
end

function split_importance(x, y; kwargs...)
    estimator = LGBMRegression()
    LightGBM.fit!(estimator, x,y)
    return LightGBM.split_importance(estimator)
end
