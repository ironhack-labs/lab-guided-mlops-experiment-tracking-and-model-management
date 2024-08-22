# 1. Experiment tracking and model management
Experiment tracking is a crucial component in Machine Learning Operations (MLOps), which is the practice of managing the lifecycle of machine learning projects. As the complexity of machine learning models increases, so does the need for systematic and reproducible experiment management. This is where experiment tracking comes into play.

**What is Experiment Tracking?**

Experiment tracking involves the systematic logging and management of all aspects related to machine learning experiments. This includes:

- **Hyperparameters:** Settings that govern the training process, such as learning rates, batch sizes, and regularization terms.
- **Datasets:** Versions of the data used for training and testing models.
- **Code Versions:** Specific versions of the codebase used for running experiments.
- **Environment Configurations:** Details about the computing environment, including hardware specifics and software dependencies.
- **Results and Metrics:** Outcomes of the experiments, such as accuracy, loss, precision, and recall.
- **Artifacts:** Models, logs, and other files generated during the experiment.

**Why is Experiment Tracking Important?**

1. **Reproducibility:** Enables researchers and data scientists to reproduce experiments and verify results by providing a detailed record of all experiment variables and outcomes.
2. **Collaboration:** Facilitates collaboration among team members by providing a centralized repository of experiment details, allowing others to understand, replicate, and build upon previous work.
3. **Optimization:** Helps in systematically comparing different experiments to optimize models by evaluating the effects of different hyperparameters, data preprocessing steps, and other variables.
4. **Compliance:** Ensures adherence to regulatory and organizational standards by maintaining detailed records of the machine learning development process.
5. **Transparency:** Provides clear visibility into the experiment lifecycle, making it easier to debug and improve models.

**Tools for Experiment Tracking**

Several tools and platforms support experiment tracking in MLOps, each offering unique features to enhance the experiment management process:

- **MLflow:** An open-source platform that provides comprehensive experiment tracking, model management, and deployment functionalities.
- **Weights & Biases:** A popular tool for tracking experiments, visualizing metrics, and collaborating with team members.
- **Neptune:** A platform designed for managing and tracking machine learning experiments, models, and data.
- **Comet.ml:** Provides experiment tracking, model management, and visualizations to aid in the development of machine learning models.

## 1.1 Experiment tracking intro
### MLflow
## Using MLflow for Experiment Tracking

MLflow is an open-source platform designed to manage the end-to-end machine learning lifecycle. It provides robust experiment tracking capabilities, allowing data scientists and researchers to log and manage all aspects of their machine learning experiments efficiently.

### Key Features of MLflow

- **Experiment Tracking:** Log parameters, metrics, and artifacts from your experiments, enabling detailed tracking and comparison.
- **Model Management:** Register and version models, facilitating easy deployment and reproducibility.
- **Deployment:** Simplifies the process of deploying machine learning models to various platforms.
- **Scalability:** Supports large-scale machine learning projects, making it suitable for both individual researchers and large organizations.

MLflow currently offers four components:

- MLflow Tracking
Record and query experiments: 
  - code 
  - data
  - config
  - results

- MLflow Projects

    If a project contains multiple algorithms that can be run separately, in that multiple entry points are mentioned in MLProject file.

    Properties of a project:
  - Name - Name of the project 
  - Entry Points - Typically a .py or .sh file to run the entire project or some specific functionality, say an algorithm. List of entry points are mentioned in MLProject file 
  - Environment - Specifications such as library dependencies for the software environment for the code to run. Supported environments - conda environments, virtualenv environments, docker environments.

- MLflow Models

    A typical model directory contains the following files:

    - MLmodel - a YAML file describing model flavours, time created, run_id if the model was created in experiment tracking, signature denoting input and output details, input example, version of databricks runtime (if used) and mlflow version 
    - model.pkl - saved model pickle file 
    - conda.yaml - environment specifications for conda environment manager 
    - python_env.yaml - environment specification for virtualenv environment manager 
    - requirements.txt - list of pip installed libraries for dependencies
- Model Registry

    Model Registry concepts to manage life cycle of mlflow model:

    - Model - An mlflow model logged with one of the flavours mlflow.<model_flavour>.log_model()
    - Registered model - An mlflow model registered on Model Registry. It has a unique name, contains versions, transitional stages, model lineage and other associated metadata. 
    - Model Version - Version of the registered model 
    - Model Stage - Each distinct model version can be associated with one stage at a time. Stages supported are Staging, Production and Archived. 
    - Annotations and descriptions - Add useful information such as descriptions, data used, methodology etc. to the registered model.

***Ref:***  https://www.mlflow.org/

## 1.2 Getting started with MLflow
***Note*** *Run in the local environment.*
### Prepare the environment ###
Run the following command to create a fresh new new conda virtual environment.  
```conda create -n exp-tracking-env python=3.9```
  
Next we activate the newly created environment.  
```conda activate exp-tracking-env```
  
Install the required packages listed in requirements.txt file.  
```pip install -r requirements.txt```

Launch mlflow ui as well. Run the following command to start mlflow ui (a gunicorn server) connected to the backend sqlite database.  
```mlflow ui --backend-store-uri sqlite:///mlflow.db```

To access mlflow ui open `https://127.0.0.1:5000` in your browser.

## Note for MLflow tracking:
An MLflow tracking server has two components for storage: a backend store and an artifact store.

The backend store is where MLflow Tracking Server stores experiment and run metadata as well as params, metrics, and tags for runs. MLflow supports two types of backend stores: file store and database-backed store.

Use `--backend-store-uri` to configure the type of backend store. You specify a file store backend as `./path_to_store` or `file:/path_to_store` and a database-backed store as SQLAlchemy database URI. 
The database URI typically takes the format `<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>`. MLflow supports the database dialects ***mysql, mssql, sqlite***, and ***postgresql***. 
Drivers are optional. If you do not specify a driver, SQLAlchemy uses a dialect’s default driver. For example, `--backend-store-uri sqlite:///mlflow.db` would use a local SQLite database.

By default `--backend-store-uri` is set to the local `./mlruns` directory (the same as when running `mlflow run` locally), but when running a server, make sure that this points to a persistent (that is, non-ephemeral) file system location.

The artifact store is a location suitable for large data (such as an S3 bucket or shared NFS file system or as our use case: HDFS ) and is where clients log their artifact output (for example, models). artifact_location is a property recorded on mlflow.entities.Experiment for default location to store artifacts for all runs in this experiment. Additional, artifact_uri is a property on mlflow.entities.RunInfo to indicate location where all artifacts for this run are stored.

Use --default-artifact-root (defaults to local ./mlruns directory) to configure default location to server’s artifact store. This will be used as artifact location for newly-created experiments that do not specify one. Once you create an experiment, --default-artifact-root is no longer relevant to that experiment.

## 1.3 Experiment tracking with MLflow

#### Linear model
In the **01_Linear Model Example Notebook**

#### Xgboost
In the **02_Xgboost example notebook**

#### Ensemble
In the **03_Ensemble example notebook**

## 1.4 Model management
* Load model as an artifact:
`mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")`

* Log model using the method `log_model`:
`mlflow.<framework>.log_model(model, artifact_path = "models/")`

Examples are in **02_Xgboost example.ipynb**

## 1.5 Model registry
[Model Registry](https://mlflow.org/docs/latest/model-registry.html) is to manage life cycle of mlflow model for improving efficiency in developing:
![model_registry.png](images%2Fmodel_registry.png)

The concepts should be known:
* Model - An mlflow model logged with one of the flavours `mlflow.<model_flavour>.log_model()`
* Registered model - An mlflow model registered on Model Registry. It has a unique name, contains versions, transitional stages, model lineage and other associated metadata. 
* Model Version - Version of the registered model 
* Model Stage - Each distinct model version can be associated with one stage at a time. Stages supported are Staging, Production and Archived. 
* Annotations and descriptions - Add useful information such as descriptions, data used, methodology etc. to the registered model.

[Two workflow](https://mlflow.org/docs/latest/model-registry.html) for it, one is from UI, the other is from API.

1). UI workflow;

2). API workflow: **04_Model Registry workflow example**

Explaining workflow:
- Adding an MLflow Model to the Model Registry

- Fetching an MLflow Model from the Model Registry

- Serving an MLflow Model from Model Registry

- Adding or Updating an MLflow Model Descriptions

- Renaming an MLflow Model

- Transitioning an MLflow Model’s Stage

- Listing and Searching MLflow Models

- Archiving an MLflow Model

- Deleting MLflow Models

- Registering a Model Saved Outside MLflow

- Registering an Unsupported Machine Learning Model

- Using Registered Model Aliases

## 1.6 MLflow in practice
Depending upon the project and number of data scientists going to collaborate, the configurational aspect of mlflow is decided. Consider the following three scenarios:

- A single data scientist participating in a competition
[scenario-1.ipynb](running-mlflow-examples%2Fscenario-1.ipynb)
- A cross-functional team with single data scientist
[scenario-2.ipynb](running-mlflow-examples%2Fscenario-2.ipynb)
- Multiple data scientists working together on models
[scenario-3.ipynb](running-mlflow-examples%2Fscenario-3.ipynb)


## 1.7 MLflow: benefits, limitations and alternatives

- Benefits
    - Share and collaborate with other members
    - More visibility into all the efforts
      
- Limitations
  - Security - restricting access to the server
  - Scalability
  - Isolation - restricting access to certain artifacts
    
- When not to use

  - Authentication and user profiling is required
  - Data versioning - no in-built functionality but there are work arounds
  - Model/Data monitoring and alerts are required
    
- Alternates
  - Nepture.ai
  - Comet.ai
  - Weights and Biases
  - etc
