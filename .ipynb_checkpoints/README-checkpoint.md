LightGBM regression model for Use Case 2. The model was trained on Italy's national 1-hour resolution load time series. 

The training set starts on 2015-04-09 and ends on 2019-12-31. The timeseries from 2020-01-01 to 2020-12-31 is considered as the validation set, and from 2021-01-01 to 2022-01-01 as the test set.

The model's parameters that were used are:
    
lags: 216 (that is the lookback window, which is the minimum size of timeseries that can be given to the model as input)
lags_past_covariates : null 
lags_future_covariates : None 
future_covs_as_tuple : True 
random_state : 0 

For more information check the darts documentation: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html

The forecast horizon used during evaluation was 24 hours, and it was performed using backtesting: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html#darts.models.forecasting.lgbm.LightGBMModel.historical_forecasts. Code used for evaluation is in the folder named model_evaluation.