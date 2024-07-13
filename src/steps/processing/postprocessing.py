import pandas as pd 
import numpy as np 
from abc import ABC, abstractmethod
from typing import Union
from sklearn.metrics import *

class ProcessStrategy(ABC):
    def postprocessing_output(self, output: Union[pd.DataFrame | np.ndarray]):
        pass 

class PostProcessingStrategy(ProcessStrategy): 
    def postprocessing_output(self, output: Union[pd.DataFrame | np.ndarray]):
        return super().postprocessing_output(output)