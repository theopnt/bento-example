This repository contains an example of model serving with bento.

After cloning the repo, you should install the requirements:
```pip install -r requirements.txt``

Then, run the notebook save_model_as_bento.ipynb to save your model from mlflow as a bento. This notebook also contains code that checks if
the model is providing predictions as expected.

Then, service.py may need some changes. This is the python file that defines the api of the model. If your model is the same type as this example
(global forecasting model for UC7), then you only need to change the model's name in the definition of the runner.

bentofile.yaml describes the container to be created for the bento. If your model is the same type as this example
(global forecasting model for UC7), then nothing needs to be changed here. 

After that, we run the serving.ipynb notebook to test the service without docker. That is in order to make sure that it is working. 
After running the final instruction of this notebook, the user must run prediction_examples.ipynb to make a prediction for the example files provided.

After the testing is complete, the user can make a docker container for the service. All the following commands may have 2 versions: One exclusivelly for jupyterlab,
and one for local machines.

We open the terminal in the folder of this repository, and run:

```/opt/anaconda3/bin/bentoml build``` (jupyterlab)

or 

```bentoml build``` (local machine)

This command builds the service in this repository. The resulting tag of the bento (here uc7_companies_no_covs:example)
will be needed for the later commands. Then, run:

```sudo su```

```export BENTOML_HOME=/new_vol_300/marija/conda/multiusers/iccs/bentoml``` (This command is needed only for jupyterlab)

```bentoml containerize uc7_companies_no_covs:example```

This command builds the docker container.

```docker run -it --rm -p 3060:3000 uc7_companies_no_covs:example```

This command runs the docker container. Now the container is running on port 3060, and the user can make predictions using prediction_examples.ipynb.
