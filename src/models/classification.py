from abc import ABC, abstractmethod
import logging
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class Model(ABC): 
    def __init__(self, *args, **kwargs): 
        pass 
    
    @abstractmethod
    def fit(self, X, y): 
        pass 
    
    @abstractmethod
    def predict(self, X): 
        pass 
    

class LogisticRegressionClassification(Model): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.model = LogisticRegression(*args, **kwargs)
    
    def fit(self, X, y): 
        try: 
            self.model.fit(X, y)
            logger.info(f"Model {self.__class__.__name__} trained successfully")
            return self.model
        except ValueError as e: 
            logger.error(f"Error: {e}")
            return None
    
    def predict(self, X): 
        return self.model.predict(X)
    

class SVM(Model): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.model = SVC(*args, **kwargs)
    
    def fit(self, X, y): 
        try:    
            self.model.fit(X, y)
            logger.info(f"Model {self.__class__.__name__} trained successfully")
            return self.model
        except ValueError as e: 
            logger.error(f"Error: {e}")
            return None
    
    def predict(self, X): 
        return self.model.predict(X)
    

class RandomForest(Model): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.model = RandomForestClassifier(*args, **kwargs)
    
    def fit(self, X, y):
        try: 
            self.model.fit(X, y)
            logger.info(f"Model {self.__class__.__name__} trained successfully")
            return self.model
        except ValueError as e: 
            logger.error(f"Error: {e}")
            return None
    
    def predict(self, X): 
        return self.model.predict(X)