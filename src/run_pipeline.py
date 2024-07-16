from pipeline.training_pipeline import training_pipeline
from pipeline.deployment_pipeline import (
    deployment_pipeline, 
    inference_pipeline
)
import click
from pipeline.steps.models.model_config import *
from zenml.client import Client 

@click.command()
@click.option(
    "--config", 
    "-c", 
    type = click.Choice(['DEPLOY', 'PREDICT', 'DEPLOY_AND_PREDICT']),
    default = 'DEPLOY_AND_PREDICT', 
    help = 'Optionally, you can choose to run the pipeline for \
            deploying (DEPLOY), predicting (PREDICT), but with the default case, \
            it will be deploy and predict (DEPLOY_AND_PREDICT)'
)

@click.option(
    "--min-accuracy", 
    default = 0.80,
    help="Minimum accuracy required to deploy the model"
)

def run_deployment(confi: str, min_accuracy):
    pass 

if __name__=='__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path=r'C:\Users\lenovo\OneDrive - Cong ty co phan Format Vietnam JSC\Desktop\simple-mlops\data\raw\telecom_churn.csv') 