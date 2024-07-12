from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters): 
    """Model Configs

    Args:
        BaseParameters (_type_): _description_
    """
    def __init__(self, model_name: str):
        self.model_name = model_name