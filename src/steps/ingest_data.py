import logging
import sklearn.datasets
from zenml import steps
import pandas as pd

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(filenamm='app.log',level=logging.INFO)

# Initiate loading data function
@steps
def ingest_data(data_path: str) -> pd.DataFrame:
    try:
        logger.info('Logging data ... ')
        df = pd.read_csv(data_path)
        logger.info('Data loaded successfully')
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

