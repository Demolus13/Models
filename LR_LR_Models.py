"""
Importing the necessary libraries
"""
import os

import torch
import torch.nn as nn

import numpy as np

# Remove all the warnings
import warnings
warnings.filterwarnings('ignore')

# Set env CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearRegression(nn.Module):
    """
    A Linear Regressor.
    """

    def __init__(self, in_features: int, out_features: int, learning_rate: float = 0.01, random_state: int = None):
        """
        Constructor for Linear Regression.

        in_features: int: The number of input features.
        out_features: int: The number of output features.
        random_state: int: The seed for the random number generator.
        """
        
        super(LinearRegression, self).__init__()
        if random_state is not None:
            np.random.seed(random_state)
        self.W = torch.tensor(np.random.randn(out_features, in_features), dtype=torch.float32).to(device)
        self.b = torch.tensor(np.random.randn(1, out_features), dtype=torch.float32).to(device)
        self.lr = learning_rate
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: torch.Tensor: The input tensor.
        """

        return torch.matmul(X, self.W.T) + self.b
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, print_cost: bool = False):
        """
        X: torch.Tensor: The input tensor.
        """

        X, y = X.reshape(-1, self.in_features).to(device), y.reshape(-1, self.out_features).to(device)

        Cost = []
        for i in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            cost = torch.mean((predictions - y) ** 2)/2
            Cost.append(cost.item())

            # Backward pass
            dW = torch.matmul((predictions - y).T, X)/X.shape[0]
            db = torch.mean(predictions - y, dim=0, keepdim=True)

            # Update the weights
            self.W = (self.W - self.lr * dW).to(device)
            self.b = (self.b - self.lr * db).to(device)

            # Print the cost
            if print_cost and (i+1) % 100 == 0:
                print(f'Cost at epoch {i+1}: {round(cost.item(), 3)}')
                print("\n------------------------------------------------------------\n")

        return Cost
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: torch.Tensor: The input tensor.
        """

        return self.forward(X)

    def save_model(self, path):
        """
        Save the model parameters.

        path: str: The path where the model parameters should be saved.
        """
        torch.save({
            'W': self.W,
            'b': self.b,
            'lr': self.lr,
            'in_features': self.in_features,
            'out_features': self.out_features
        }, path)

    @staticmethod
    def load_model(path):
        """
        Load the model parameters.

        path: str: The path from where the model parameters should be loaded.
        """
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model = LinearRegression(in_features=state_dict['in_features'], out_features=state_dict['out_features'], learning_rate=state_dict['lr'])
        model.W = state_dict['W']
        model.b = state_dict['b']
        return model