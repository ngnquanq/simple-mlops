from pipeline.training_pipeline import training_pipeline
from pipeline.deployment_pipeline import (
    continuos_deployment_pipeline
)
import click
from pipeline.steps.models.model_config import *
from zenml.client import Client 

from typing import cast
from rich import print 
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import logging

logger = logging.getLogger(__name__)

DEPLOY = 'deploy'
PREDICT = 'predict'
DEPLOY_AND_PREDICT = 'deploy_and_predict'

@click.command()
@click.option(
    "--config", 
    "-c", 
    type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default = DEPLOY_AND_PREDICT, 
    help = 'Optionally, you can choose to run the pipeline for \
            deploying (DEPLOY), predicting (PREDICT), but with the default case, \
            it will be deploy and predict (DEPLOY_AND_PREDICT)'
)

@click.option(
    "--min-accuracy", 
    default = 0.80,
    help="Minimum accuracy required to deploy the model"
)

def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    if deploy: 
        continuos_deployment_pipeline(min_accuracy = min_accuracy, 
                                      workers = 3,
                                      timeout = 60,)
    if predict:
        pass
    
    print(
        "You can run: \n" 
        f"[italic green] mlflow ui --backend-store-uri '{get_tracking_uri()}"
        "[/italic green]\n ...to inspect your experiment runs within the MLflow"
        "UI.\nYou can find your runs tracked within the "
        "`mlflow_example_pipeline` experiment. There you'll also be able to "
        "compare two or more runs. \n\n"
    )
    
    # Fetch existing services with the same pipeline: name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",  
        pipeline_step_name='mlflow_model_deployer_step',
        model_name='model'
    )
    
    if existing_services:
        logger.info('Exist services inside the local machine')
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLFlow prediction server is running locally as a daemon"
                f"process service and accepts inference requests at: \n"
                f"  {service.prediction_url}\n "
                f"To stop the service, run"
                f"[italic green]` zenml model-deployer models delete"
                f"{str(service.uuid)}`[/italic green]"
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state: \n"
                f"->    Last state: '{service.status.state.value}'"
                f"->    Last error: `{service.status.last_error}`"
            )
        else:
            print(
                "No mlflow prediction server is currently running. The deployment "\
                    "pipeline must run first to train a model and deploy it. Execute "\
                        "the same command with the `--deploy` argument to deploy a model "
            )
            

if __name__=='__main__': 
    run_deployment()