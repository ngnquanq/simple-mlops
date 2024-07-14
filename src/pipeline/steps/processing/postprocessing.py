import pandas as pd 
import numpy as np 
from abc import ABC, abstractmethod
from sklearn.metrics import *

class Evaluate(ABC):
    """
    Use this class to output the result from the predict.
    """
    @abstractmethod
    def evaluate_metric(self, metric: str):
        pass 