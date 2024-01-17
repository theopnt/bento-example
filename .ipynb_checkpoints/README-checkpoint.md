This repository contains an example of model serving with bento.

To serve your model you must first have bento installed in your system.

Run the notebook save_model_as_bento.ipynb to save your model from mlflow as a bento. This notebook also contains code that checks if
the model is providing predictions as expected.

Then, service.py may need some changes. This is the python file that defines the api of the model. If your model is the same type as this example
(global forecasting model for UC7), then you only need to change the model's name in the definition of the runner.

bentofile.yaml describes the container to be created for the bento. If your model is the same type as this example
(global forecasting model for UC7), then nothing needs to be changed here. 

After that, we run the serving notebook to test the service without docker. That is in order to check that it is working. After the user
has run the final instruction of this notebook, they must run prediction_examples.ipynb to make a prediction for the example files provided.

After all our testing, we can make a docker container for our service. All the following commands have been tested in the enviroment of jupyterlab.
If you run this on your machine, you have to use the second command. 

We open the terminal in the folder of this repository, and run:

```/opt/anaconda3/bin/bentoml build``` (jupyterlab)

or 

```bentoml build``` (local machine)

This command builds the service in this repository. The resulting tag of the bento (here uc7_companies_no_covs:example)
will be needed for the later commands. We then run:

```sudo su```

```export BENTOML_HOME=/new_vol_300/marija/conda/multiusers/iccs/bentoml``` (This command is needed only for jupyterlab)

```bentoml containerize uc7_companies_no_covs:example```

This command builds the docker container.

```docker run -it --rm -p 3060:3000 uc7_companies_no_covs:example```

This command runs the docker container.