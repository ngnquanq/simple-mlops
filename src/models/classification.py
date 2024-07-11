from abc import ABC, abstractmethod
from sklearn.base import ClassifierMixin

class Model(ABC): 
    def __init__(self, *args, **kwargs): 
        pass 
    
    @abstractmethod
    def fit(self, X, y): 
        pass 
    
    @abstractmethod
    def predict(self, X): 
        pass 
    
