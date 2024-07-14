from zenml import step
import pandas as pd
import logging 
from typing import Union, Tuple, Annotated
from .models.evaluate_model import *
from sklearn.base import BaseEstimator, ClassifierMixin

logging.basicConfig(filename='logs/evaluate.log')
logger = logging.getLogger(__name__)

@step
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
    
    return cm, metrics