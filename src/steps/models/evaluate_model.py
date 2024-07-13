import logging
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from typing import Dict 
import pandas as pd 
import typing


logging.basicConfig(filename='logs/Evaluate.log')
logger = logging.getLogger(__name__)


class Evaluation(ABC): 
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass 
    

class ConfusionMatrix(Evaluation):
    """Evaluattion Strategy that used Confusion Matrix
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        try: 
            logger.info('Calculating Confusion matrix')
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            logger.info('Calculate confusion matrix complete')
            return cm
        except Exception as e: 
            logging.error("Error in calculating Confusion matrix: {}".format(e))
            raise e 
        

class ClassificationMetrics(Evaluation):
    """ Evaluate strategy that uses other metrics like AUC, accuracy, etc. """
    
    def calculate_score(self, y_true: typing.Union[np.ndarray, pd.Series, pd.DataFrame], 
                        y_pred: typing.Union[np.ndarray, pd.Series, pd.DataFrame]) -> Dict[str, float]:
        try:
            # Convert DataFrame or Series to NumPy array
            if isinstance(y_true, (pd.DataFrame, pd.Series)):
                y_true = y_true.to_numpy().ravel()
            if isinstance(y_pred, (pd.DataFrame, pd.Series)):
                y_pred = y_pred.to_numpy().ravel()
            
            logger.info('Calculating other metrics')
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            logger.info('Calculate confusion matrix complete')
            return {
                'accuracy': accuracy, 
                'f1': f1, 
                'recall': recall, 
                'precision': precision
            }
        except Exception as e:
            logger.error("Error in calculating other metrics: {}".format(e))
            raise e