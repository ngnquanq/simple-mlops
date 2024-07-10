import logging
from zenml import step
import pandas as pd

@step
def train_model(df: pd.DataFrame) -> None: 
    pass