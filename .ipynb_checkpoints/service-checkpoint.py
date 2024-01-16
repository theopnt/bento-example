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
lgbm_italy_runner = bentoml.mlflow.get("lgbm_italy:b2xpt4wsbw7m76qw").to_runner()

# Create the iris_classifier service with the ScikitLearn runner
# Multiple runners may be specified if needed in the runners array
# When packaged as a bento, the runners here will included
lgbm_italy_service = bentoml.Service("lgbm_italy", runners=[lgbm_italy_runner])

input_spec = Multipart(n=Text(), 
                       series_uri=Text(),
                       roll_size=Text(),
                       future_covariates_uri=Text(),
                       past_covariates_uri=Text(),
                       batch_size=Text())

@lgbm_italy_service.api(input=input_spec, output=Text())
def predict(n : str, 
            series_uri : str,
            roll_size : str,
            future_covariates_uri : str,
            past_covariates_uri : str,
            batch_size : str) -> pd.DataFrame:
    try:
        n = int(n)        
    except:
        raise BentoMLException("Can't convert parameters to int")
            
    model_input = {
        "n": n,
        "series_uri": series_uri,
        "roll_size": roll_size,
        "future_covariates_uri": future_covariates_uri,
        "past_covariates_uri": past_covariates_uri,
        "batch_size": batch_size}
    print(model_input)
    res = lgbm_italy_runner.predict.run([model_input])
    return str(res[0].to_dict())