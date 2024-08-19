import shap
import lime 
from lime.lime_tabular import LimeTabularExplainer
import logging
import pandas as pd
import numpy as np 
from zenml import step
from typing import Dict, Union
from .models.model_config import *
from .models.classification import *
from sklearn.base import ClassifierMixin
import mlflow 
from zenml.client import Client 
import json

logging.basicConfig(filename='logs/XAI.log')
logger = logging.getLogger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def shap_explained(
    trained_models: list[ClassifierMixin],
    train_data: Union[pd.DataFrame, np.ndarray],
    test_data: Union[pd.DataFrame, np.ndarray]
) -> Dict[str, shap.explainers._tree.TreeExplainer]:
    
    # Fitting the explainers
    explainers = {}
    for model in trained_models:
        explainer = shap.Explainer(model, train_data)
        explainers[model.__class__.__name__] = explainer 
        
    # Calculate the SHAP value
    shap_values_test = {}
    for model_name, explainer in explainers.items():
        shap_values_test[model_name] = explainer(test_data, check_additivity=False)
    
    # Save the shap values in a json file
    shap_values_test_serialized = {model_name: shap_values_test[model_name].values.tolist() for model_name in shap_values_test.keys()}
    with open(r'C:\Users\lenovo\OneDrive - Cong ty co phan Format Vietnam JSC\Desktop\simple-mlops\reports\shap_values_test.json', 'w') as f:
        json.dump(shap_values_test_serialized, f)
        
    return explainers

@step(experiment_tracker=experiment_tracker.name)
def lime_explained(
    training_data, testing_data
    feature_names, class_names, 
    trained_models
):
    lime_explainer = LimeTabularExplainer(
        training_data=training_data,
        feature_names=training_data.columns,
        class_names=['Churn_0.0', 'Churn_1.0']
        verbose=True, discretize_continuous=True,
        random_state=1
    )
    explaination_dict = {}
    indices = []
    
    