from zenml import step
import pandas as pd
import logging 

@step
def evaluate_model(df: pd.DataFrame) -> None: 
    pass
