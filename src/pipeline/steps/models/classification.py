from abc import ABC, abstractmethod
import logging
import optuna
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

logging.basicConfig(filename='logs/createModel.log')
logger = logging.getLogger(__name__)

# ================================================ Machine Learning Model with SKLearn =======================================
class Model(ABC): 
    def __init__(self, *args, **kwargs): 
        pass 
    
    @abstractmethod
    def fit(self, X, y): 
        pass 
    
    @abstractmethod
    def predict(self, X): 
        pass 
    
    @abstractmethod
    def optimize(self, trial: optuna.Trial, X_train, y_train, X_test, y_test): 
        """Optimize the hyperparameter of the model[]

        Args:
            trial (Optuna Trial object): Optuna Trial object
            X_train (ndarray): Training feature
            y_train (ndarray): Training target
            X_test (ndarray): Testing feature
            y_test (ndarray): Testing target
            
        """
        pass

class LogisticRegressionClassification(Model): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.model = LogisticRegression(*args, **kwargs)
    
    def fit(self, X, y) -> ClassifierMixin:  
        try: 
            self.model.fit(X, y)
            logger.info(f"Model {self.__class__.__name__} trained successfully")
            return self.model
        except ValueError as e: 
            logger.error(f"Error: {e}")
            return None
    
    def predict(self, X): 
        return self.model.predict(X)
    
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        return super().optimize(trial, X_train, y_train, X_test, y_test)
    

class SVM(Model): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.model = SVC(*args, **kwargs)
    
    def fit(self, X, y) -> ClassifierMixin: 
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
    
    def fit(self, X, y) -> ClassifierMixin:
        try: 
            self.model.fit(X, y)
            logger.info(f"Model {self.__class__.__name__} trained successfully")
            return self.model
        except ValueError as e: 
            logger.error(f"Error: {e}")
            return None
    
    def predict(self, X): 
        return self.model.predict(X)
    
    def optimize(self, trial: optuna.Trial, X_train, y_train, X_test, y_test):
        n_estimators = trial.suggest_int("")
        

# =============================================== Deep Learning Model with PyTorch ============================================
class ClassificationModel(ABC, nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        """
        pass

    @abstractmethod
    def train_model(self, dataloader, criterion, optimizer, num_epochs):
        """
        Method to train the model.
        """
        pass

    @abstractmethod
    def evaluate_model(self, dataloader, criterion):
        """
        Method to evaluate the model.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Method to make predictions.
        """
        pass

    def save_model(self, path):
        """
        Save the model to a file.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        Load the model from a file.
        """
        self.load_state_dict(torch.load(path))


# ================================================== Parameter Tuning =========================================================
class HyperparameterTuner: 
    """
    Class for perming hyperparameter tuning using Optuna based on Model Strategy
    """
    
    def __init__(self, model: Model, X_train, y_train, X_test, y_test ):
        self.model = model 
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def optimize(self, n_trials = 100): 
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.model.optimize(trial, self.X_train, self.y_train, 
                                                         self.X_test, self.y_test), 
                                                         n_trials = n_trials)
        return study.best_trial.params

# =============================== PyTorch Deep Learning Model ====================================
class GraphClassification(nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers, *args, **kwargs):
        super(GraphClassification, self).__init__(*args, **kwargs)
        
        # Define multiple GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(1, num_gcn_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            
        # Define multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    # Calculate edge index
    def forward(self, x, edge_index):
        pass
            