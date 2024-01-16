# service.py
import pandas as pd
import bentoml
#import scikit-learn
from bentoml import mlflow
from bentoml.io import PandasDataFrame
from bentoml.io import Text
from bentoml.io import Multipart
from bentoml.exceptions import BentoMLException
     
# Load the runner for the latest model we just saved
lgbm_uc6_w6_pos_ac_runner = bentoml.mlflow.get("lgbm_uc6_w6_pos_ac:latest").to_runner()

# Create the iris_classifier service with the ScikitLearn runner
# Multiple runners may be specified if needed in the runners array
# When packaged as a bento, the runners here will included
lgbm_uc6_w6_pos_ac_service = bentoml.Service("lgbm_uc6_w6_pos_ac", runners=[lgbm_uc6_w6_pos_ac_runner])

input_spec = Multipart(n=Text(), 
                       series=PandasDataFrame())

@lgbm_uc6_w6_pos_ac_service.api(input=input_spec, output=Text())
def predict(n : str, 
            series : pd.DataFrame,
            ) -> pd.DataFrame:
    try:
        n = int(n)        
    except:
        raise BentoMLException("Can't convert parameters to int")            
    model_input = {
        "n": n,
        "history": series,
        "series_uri": "",
        "roll_size": "24",
        "future_covariates_uri": "None",
        "past_covariates_uri": "None",
        "batch_size": 1}
    print(series)
    res = lgbm_uc6_w6_pos_ac_runner.predict.run([model_input])
    return res[0].to_json()