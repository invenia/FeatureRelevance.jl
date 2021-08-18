struct PredictivePowerScore <: FeatureRelevanceCriterion end

function (::PredictivePowerScore)(x, y; kwargs...)
    return pps(x, y; kwargs...)
end

function pps(x,y; n_folds=4, kwargs...)
    # regression task error metric variables
    sum_ae_model = 0.0
    sum_ae_naive = 0.0

    num_obs = length(x)
    segment_length = floor(num_obs/n_folds)
    fold_idxs = Int64.(floor.(collect((1:num_obs) ./ segment_length)) .+ 1)
    # last fold may be longer than previous due to inability to inability to equally split
    fold_idxs[fold_idxs .> n_folds] .= n_folds

    num_preds = 0
    for fold_num in 1:n_folds
        train_x, train_y, test_x, test_y = _prepare_cv_fold_data(x, y, fold_idxs, fold_num)

	estimator = LGBMRegression()
	LightGBM.fit!(estimator, train_x, train_y)
	preds = LightGBM.predict(estimator, test_x)

        sum_ae_model += sum(abs.(preds .- test_y))
        sum_ae_naive += sum(abs.(median(train_y) .- test_y))
        num_preds += length(test_y)
    end

    mae_model = sum_ae_model / num_preds
    mae_naive = sum_ae_naive / num_preds
    if mae_model > mae_naive
        pps = 0 # this avoids negative pps values
    else
        pps = 1 - (mae_model / mae_naive)
    end
    return pps
end

function _prepare_cv_fold_data(x, y, fold_idxs, fold_num)
    # train on everything except the one segment
    train_idxs = findall(fold_idxs .!= fold_num)
    test_idxs = findall(fold_idxs .== fold_num)

    train_x = hcat(x[train_idxs])
    train_y = y[train_idxs]

    test_x = hcat(x[test_idxs])
    test_y = y[test_idxs]

    return train_x, train_y, test_x, test_y
end
