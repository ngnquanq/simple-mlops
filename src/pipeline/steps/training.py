import logging
import pandas as pd
import numpy as np 
from zenml import step
from .models.model_config import *
from .models.classification import *
from sklearn.base import ClassifierMixin

logging.basicConfig(filename='logs/training.log')
logger = logging.getLogger(__name__)

@step
def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    config: ModelNameConfig) -> ClassifierMixin: 
    
    model = None
    if config.name == 'LogisticRegression':
        model = LogisticRegressionClassification()
    elif config.name =='RandomForest': 
        model = RandomForestClassifier()
    elif config.name == 'SVM':
        model = SVM()
    logger.info('The selected model is: {}'.format(config.name))
    
    trained_model = model.fit(X_train, y_train)
    logger.info('Training the model completed.')
    
    return trained_model