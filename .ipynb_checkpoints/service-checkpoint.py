# service.py
import pandas as pd
import bentoml
from bentoml import mlflow
from bentoml.io import PandasDataFrame
from bentoml.io import Text
from bentoml.io import Multipart
from bentoml.exceptions import BentoMLException
     
# Load the runner for the latest model we just saved
model_runner = bentoml.mlflow.get("uc7_companies_no_covs:latest").to_runner()

#creating the service for the model. We can name it whatever we wish
model_service = bentoml.Service("uc7_companies_no_covs", runners=[model_runner])

input_spec = Multipart(n=Text(), 
                       meter_id=Text(),
                       series=PandasDataFrame())
#input must be multipart, as it contains both strings and dataframes. Dataframes are converted from json
#to dataframe by bentoml during the api call
@model_service.api(input=input_spec, output=Text())
def predict(n : str, 
            meter_id: str,
            series : pd.DataFrame,
            ) -> pd.DataFrame:
    try:
        n = int(n)        
    except:
        raise BentoMLException("Can't convert parameters to int")            
    model_input = {
        "multiple": "False",
        "weather_covariates":[],
        "resolution": "60",
        "ts_id_pred": meter_id,        
        "n": n,
        "history": series,
        "series_uri": "",
        "roll_size": "1",
        "future_covariates_uri": "None",
        "past_covariates_uri": "None",
        "batch_size": 1}
    print(series)
    res = model_runner.predict.run([model_input])
    return res[0].to_json()