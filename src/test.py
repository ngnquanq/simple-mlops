from processing.preprocessing import * 
from steps.clean_data import * 
import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt 


if __name__=='__main__':
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname,'../data/raw/telecom_churn.csv')
    df = pd.read_csv(data_path)
    X_train, X_valid, y_train, y_valid = clean_data(df)
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)