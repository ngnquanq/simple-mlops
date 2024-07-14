import pandas as pd 
import numpy as np  
import logging 
from zenml import step 
from typing import Tuple, Annotated
from .processing.preprocessing import DataPreProcessStrategy, DataDivideStrategy, DataCleaning

@step()
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_valid'],
    Annotated[pd.DataFrame, 'y_train'],
    Annotated[pd.DataFrame, 'y_valid']
]:
    try: 
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        cleaned_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_dividing = DataCleaning(cleaned_data, divide_strategy)
        dataframe = data_dividing.handle_data()
        return dataframe
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise e 

if __name__=='__main__':
    df = pd.read_csv('../../data/telecom_churn.csv')
    X_train, X_valid, y_train, y_valid = clean_data(df)
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)