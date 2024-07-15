import logging
from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd 
from typing import Union, Tuple, Annotated
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Kinda useless ngl ._.
# logging.basicConfig(filename='preprocessing.log',
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logging.DEBUG)

logger = logging.getLogger(__name__)


# Abstract class is a strategy for handling data
class DataStrategy(ABC):
    """Abstract class defining strategy for handling data

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass 
    
    
class DataPreProcessStrategy(DataStrategy):
    """Inherit the datastrategy and overwrite the handle_data method provided by the DataStrategy above"""
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame: 
        """Preprocess the dataframe

        Args:
            data (pd.DataFrame): DataFrame that need to be preprocessed.
        """
        # Drop useless coloumns
        logger.info("Begin to preprocessing the dataframe ...")
        try: 
            logger.info("1. Start dropping useless columns")
            data = data.drop(columns=[
                "Account length", 
                "State", 
                "Area code"
            ])
            logger.info("Delete useless columns complete")
        except Exception as e:
            logger.exception(f"Encounting an exception when dropping columns")
            raise e
        
        # Convert data type.
        try: 
            # Converting object column to category
            logging.info("2. Converting data to it correct data types")
            for i in data.select_dtypes(include='object').columns.to_list(): 
                data[i] = data[i].astype('category')
            # Converting target column to category 
            data['Churn'] = data['Churn'].astype('category')
            logging.info("Converting data type complete")
        except Exception as e:
            logger.exception(f"Encounting an exception when convert data type")
            raise e
        
        # Handling null value.
        try: 
            if data.isnull().sum().any()==True:
                logging.info ("3. Handling null values")   
                for i in data.select_dtypes(include=['int64', 'float64']).columns.to_list():
                    data[i].fillna(data[i].mean(), inplace=True)
                data = data.dropna()
            else: 
                logging.info("3. The data had no missing values")
        except Exception as e: 
            logger.exception(f"Encounting an exception when handling null value")
        
        # Scale if needed. 
        # Identify numerical columns
        logging.info('4. Encoding the values')
        num_col = data.select_dtypes(include=['int64', 'float64']).columns.values.tolist()

        # Identify categorical columns 
        cat_col = data.select_dtypes(include='category').columns.values.tolist()

        # Encoding the data
        numeric_transformer = Pipeline(
            steps=[("Scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[('OneHotEncoder', OneHotEncoder(handle_unknown='ignore'))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_col),
                ('cat', categorical_transformer, cat_col)
            ]
        )

        encoded_data = preprocessor.fit_transform(data)
        
        # Create new names
        # Keep OG names for num_col
        new_num_col = num_col
        
        # Change name for cat_col 
        new_cat_col = preprocessor.named_transformers_['cat'].named_steps['OneHotEncoder'].get_feature_names_out(cat_col)
        
        # Combine to have new col names
        columns = list(new_num_col) + list(new_cat_col)
        
        encoded_data = pd.DataFrame(encoded_data, columns=columns)
        return encoded_data
    

class DataDivideStrategy(DataStrategy): 
    """Split the data into the dataframes for training and testing process. 

    Args:
        Dataframe (pd.DataFrame): Take in the already encoded dataframe
    """
    
    def handle_data(self, data) -> Tuple[Annotated[pd.DataFrame, 'X_train'],
                                         Annotated[pd.DataFrame, 'X_test'],
                                         Annotated[pd.DataFrame, 'y_train'],
                                         Annotated[pd.DataFrame, 'y_test']]:
        data = pd.DataFrame(data)
        X = data.iloc[:, :-2]  # All columns except the last two
        y = data.iloc[:, -2:]  # The last two columns

        # Convert one-hot encoded y to label encoding
        y = y.idxmax(axis=1).to_frame(name='Churn_0.0')
        
        # Splitting the data using train test split
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        

        return X_train, X_valid, y_train, y_valid
    
class DataCleaning(DataStrategy):
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self):
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)
    
