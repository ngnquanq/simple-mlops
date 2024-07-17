from .steps.models.model_config import *
from .steps.models import *
from .steps.ingest_data import ingest_data
from .steps.clean_data import clean_data
from .steps.evaluate import evaluate_model
from .steps.training import train_model

import logging
import pandas as pd
import numpy as np 

from zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer, )
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

# Setting the logging
logger = logging.getLogger(__name__)

# Setting basic docker and data path
docker_settings = DockerSettings(required_integrations=[MLFLOW])
data_path = r'/home/ngnqaq/project/simple-mlops/data/telecom_churn.csv'


# Adjust the Deployment Trigger Config
class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float=0.5
    
    
# Setting a step in the pipeline to get the metric out for deployment trigger
@step 
def get_metrics(target: dict ,metric: str = 'accuracy'):
    return target[metric]
    
    
# Creating the deployment trigger step to use inside the pipeline
@step
def deployment_trigger(
    accuracy: float, 
    config: DeploymentTriggerConfig,
):
    """Implement a simple deployment model trigger based on the input model accuracy
    . Then decide if the model is good enough or not. 

    Args:
        accuracy (float): _description_
        config (DeploymentTriggerConfig): _description_
    """
    return accuracy >= config.min_accuracy


# Creating a MLflowDeploymentLoaderStepParameters class with zenml BaseParameters
class MLFlowDeploymenyLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction server
        step_name: the name of the step that deployed the mlflow prediction
        running: use this tag if want to returns a running service
        model_name: the name of the model that is deployed
    """
    pipeline_name: str
    step_name: str
    running: bool = True


# Creating the pipeline for CD
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuos_deployment_pipeline(
    min_accuracy: float = 0.99,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    data = ingest_data(data_path=data_path)
    X_train, X_valid, y_train, y_valid = clean_data(data)
    trained_model = train_model(X_train = X_train, y_train = y_train, config = 'LogisticRegression')
    _, metrics = evaluate_model(trained_model, X_valid, y_valid)
    
    # Import the get_metrics step
    model_acc = get_metrics(target=metrics, metric='accuracy')
    # Create the deployment trigger
    logger.info('Making deployment decision')
    deployment_decision = deployment_trigger(accuracy = model_acc)
    logger.info('Deployment decision completed')
    mlflow_model_deployer_step(
        model = trained_model, 
        deploy_decision = deployment_decision,
        workers = workers, 
        timeout = timeout
    )
    