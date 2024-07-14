import logging
import sklearn.datasets
from zenml import step
import pandas as pd

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(filenamm='app.log',level=logging.INFO)

class IngestData:
    def __init__(self, data_path:str) -> None:
        self.data_path = data_path
    
    def get_data(self):
        logging.info(f"Ingesting data from: {self.data_path}")
        return pd.read_csv(self.data_path)

# Initiate loading data function
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """Ingesting the data from the data_path

    Args:
        data_path (str): Path to the dataframe

    Raises:
        e: Error occuring when can't read the path

    Returns:
        pd.DataFrame: Dataframe used for training model
    """
    try:
        logger.info('Loading data ... ')
        ingest_data = IngestData(data_path=data_path)
        df = ingest_data.get_data()
        logger.info('Data loaded successfully')
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

