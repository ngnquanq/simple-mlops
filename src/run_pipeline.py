from pipeline.training_pipeline import training_pipeline
from pipeline.steps.models.model_config import *
from zenml.client import Client 


if __name__=='__main__':
    training_pipeline(data_path=r'C:\Users\lenovo\OneDrive - Cong ty co phan Format Vietnam JSC\Desktop\simple-mlops\data\raw\telecom_churn.csv')