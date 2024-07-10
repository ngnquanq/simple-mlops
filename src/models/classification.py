import torch.nn as nn
import numpy as np 
import torch.optim as optim
import torch.nn.functional as F

# Initialize the model
class ClassificationModel(nn.Module):
    """For the sake of simplicity, we'll be using a simple MLP for the classification model

    Args:
        nn (torch.nn.Module): The base class for all neural network modules
        input_dim (int): The number of input features
        hidden_dim (int): The number of hidden layers
        output_dim (int): The number of output features
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        self.relu = F.relu
        self.sigmoid = F.sigmoid

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x