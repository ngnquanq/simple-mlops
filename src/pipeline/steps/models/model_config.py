from zenml.steps import BaseParameters
from .classification import *

class ModelNameConfig(BaseParameters): 
    """
    Model Configs
    """
    name: str