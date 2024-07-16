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

docker_settings = DockerSettings(required_integrations=[MLFLOW])
data_path = '../../data/telecom_churn.csv'

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float=0.8
    
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

@pipeline(enable_cache=True, 
          settings={"docker_settings": docker_settings})
def continuos_deployment_pipeline(
    min_accuracy: float = 0.8,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    data = ingest_data(data_path=data_path)
    X_train, X_valid, y_train, y_valid = clean_data(data)
    trained_model = train_model(X_train = X_train, y_train = y_train, config = 'LogisticRegression')
    _, metrics = evaluate_model(trained_model, X_valid, y_valid)
    
    # Create the deployment trigger
    deployment_decision = deployment_trigger(accuracy = metrics['accuracy'])
    mlflow_model_deployer_step(
        model = trained_model, 
        deployment_decision = deployment_decision,
        workers = workers, 
        timeout = timeout
    )
    