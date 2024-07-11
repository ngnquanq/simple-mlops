import pandas as pd 
import numpy as np 
import pathlib 
import logging 
from zenml import step 
from processing.preprocessing import DataPreProcessStrategy, DataDivideStrategy, DataCleaning

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try: 
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        cleaned_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_dividing = DataCleaning(cleaned_data, divide_strategy)
        X_train, X_valid, y_train, y_valid = data_dividing.handle_data()
        return X_train, X_valid, y_train, y_valid 
        
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise e 

if __name__=='__main__':
    df = pd.read_csv('../../data/telecom_churn.csv')
    X_train, X_valid, y_train, y_valid = clean_data(df)
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)