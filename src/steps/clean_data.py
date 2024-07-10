import pandas as pd 
import numpy as np 
import pathlib 
import logging 
from zenml import step 

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df
    