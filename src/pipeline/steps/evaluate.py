from zenml import step
import pandas as pd
import logging 
from typing import Union, Tuple, Annotated
from .models.evaluate_model import *
from sklearn.base import BaseEstimator, ClassifierMixin
import mlflow
from zenml.client import Client 

logging.basicConfig(filename='logs/evaluate.log')
logger = logging.getLogger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: ClassifierMixin,
    feature: Union[pd.DataFrame | pd.Series | np.ndarray],
    target: Union[pd.DataFrame | pd.Series] | np.ndarray
    ) -> Tuple[
        Annotated[dict, "Confusion matrix"] ,
        Annotated[dict, "Other metrics"]
        ]: 
    prediction = model.predict(feature)
    cm_class = ConfusionMatrix()
    cm = cm_class.calculate_score(y_true=target, y_pred=prediction)

    metric_class = ClassificationMetrics()
    metrics = metric_class.calculate_score(y_true=target, y_pred=prediction)
    mlflow.log_metrics(**metrics)
    
    return cm, metrics