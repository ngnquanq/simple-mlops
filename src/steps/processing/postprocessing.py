import pandas as pd 
import numpy as np 
from abc import ABC, abstractmethod

class Evaluate(ABC):
    """
    Use this class to output the result from the predict.
    """
    