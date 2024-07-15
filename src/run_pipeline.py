from pipeline.training_pipeline import training_pipeline
from pipeline.steps.models.model_config import *

if __name__=='__main__':
    training_pipeline(data_path=r'C:\Users\84898\Desktop\simple-mlops\data\telecom_churn.csv')