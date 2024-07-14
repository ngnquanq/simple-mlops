from zenml import step, pipeline
from .steps.models.model_config import *
from .steps.ingest_data import ingest_data
from .steps.clean_data import clean_data
from .steps.evaluate import evaluate_model
from .steps.training import train_model
import logging
import pandas as pd
import numpy as np 

@pipeline(enable_cache=True)
def training_pipeline(data_path: str):
    data = ingest_data(data_path=data_path)
    X_train, X_valid, y_train, y_valid = clean_data(data)
    trained_model = train_model(X_train = X_train, y_train = y_train, config = ModelNameConfig(name='LogisticRegression'))
    cm, metrics = evaluate_model(trained_model, X_valid, y_valid)
    return cm, metrics
    
if __name__=='__main__':
    training_pipeline()