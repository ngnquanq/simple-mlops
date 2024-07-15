import logging
import pandas as pd
import numpy as np 
from zenml import step
from typing import Dict
from .models.model_config import *
from .models.classification import *
from sklearn.base import ClassifierMixin
import mlflow 
from zenml.client import Client 

logging.basicConfig(filename='logs/training.log')
logger = logging.getLogger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    config: str,) -> ClassifierMixin: 
    
    mlflow.sklearn.autolog()

    model = None
    if config == 'LogisticRegression':
        model = LogisticRegressionClassification()
    elif config =='RandomForest': 
        model = RandomForestClassifier()
    elif config == 'SVM':
        model = SVM()
    logger.info('The selected model is: {}'.format(config))
    
    trained_model = model.fit(X_train, y_train)
        
    logger.info('Training the model completed.')
    
    return trained_model