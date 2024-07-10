from zenml import step, pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.evaluate import evaluate_model
from steps.training import train_model
import logging
import pandas as pd
import numpy as np 

@pipeline
def training_pipeline(data_path: str):
    data = ingest_data(data_path=data_path)
    cleaned_data = clean_data(data)
    train_model(cleaned_data)
    evaluate_model(cleaned_data)
    
if __name__=='__main__':
    training_pipeline()