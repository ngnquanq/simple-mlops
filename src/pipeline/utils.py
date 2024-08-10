import logging
import pandas as pd
from .steps.processing.preprocessing import * 

def get_data_for_test():
    try: 
        data = pd.read_csv('./data/telecom_churn.csv')
        data = data.sample(n=5, random_state=0)
        preprocess_strategy = DataPreProcessStrategy
        data_cleaning = DataCleaning(data, preprocess_strategy)
        data = data_cleaning.handle_data()
        data.drop(['Churn_0.0', 'Churn_1.0'], axis=1, inplace=True)
        result = data.to_json(orient="split")
        return result
    except Exception as e: 
        logging.error(e)
        raise e
    
def get_data_columns():
    try:
        data = pd.read_csv('./data/telecom_churn.csv')
        data = data.sample(n=5, random_state=0)
        preprocess_strategy = DataPreProcessStrategy
        data_cleaning = DataCleaning(data, preprocess_strategy)
        data = data_cleaning.handle_data()
        data.drop(['Churn_0.0', 'Churn_1.0'], axis=1, inplace=True)
        return data.columns.tolist()
    except Exception as e: 
        logging.error(e)
        raise e