"""
Importing the necessary libraries
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Remove all the warnings
import warnings
warnings.filterwarnings('ignore')

# Set env CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    """
    A Multi-Layer Perceptron.
    """

    def __init__(self, block_size: int, vocab_size: int, emb_dim: int, random_state: int = None):
        """
        Constructor for Multi-Layer Perceptron.

        block_size: int: input block size
        vocab_size: int: vocabulary of the embedded words
        emd_dim: int: embedding dimension of the characters
        random_state: int: random state for reproducibility
        """

        super(MLP, self).__init__()
        if random_state is not None:
            torch.manual_seed(random_state)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embeddings = nn.Sequential(
            nn.Embedding(vocab_size, emb_dim),
            nn.Flatten()
        )
        self.layers = nn.Sequential(
            nn.Linear(block_size * emb_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 32),
            nn.SiLU(),
            nn.Linear(32, vocab_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor: The input tensor.
        """

        x = self.embeddings(x)
        x = self.layers(x)
        return x

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 1000, batch_size: int = 4096, learning_rate: float = 0.01, print_cost: bool = False):
        """
        X: torch.Tensor: The input tensor
        y: torch.Tensor: The target tensor
        epochs: int: The number of epochs
        batch_size: int: The batch size while applying mini-batch gradient descent
        learning_rate: float: learning rate of the optimizer
        print_cost: bool: Whether to print the cost or not
        """
        self.lr = learning_rate

        X, y = X.reshape(-1, self.block_size).to(device), y.reshape(-1).to(device)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        Losses = []
        for i in range(epochs):
            for batch_X, batch_y in dataloader:
                # Forward pass
                predictions = self.forward(batch_X)
                loss = criterion(predictions, batch_y)
                Losses.append(loss.item())

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Print the cost
            if print_cost and (i+1) % 10 == 0:
                print(f'Loss at epoch {i+1}: {loss.item():.3f}')
                print("\n------------------------------------------------------------\n")

        return Losses

    def predict(self, X: torch.Tensor, decodings: dict, context_len: int):
        """
        X: torch.Tensor: The input tensor
        decodings: dict: The dictionary containing decoding of the characters
        context_len: int: The length of the context
        """

        X = X.reshape(1, self.block_size).to(device)

        i = 0
        while i <= context_len or decode != '.':
            y_pred = self.forward(X)
            id_pred = torch.distributions.Categorical(logits=y_pred).sample().item()
            decode = decodings[id_pred]
            X = torch.cat((X[:, 1:], torch.tensor([[id_pred]], device=device)), 1)
            i += 1
            yield decode

    def save_model(self, path):
        """
        Save the model parameters.

        path: str: The path where the model parameters should be saved.
        """

        model_info = {
            'block_size': self.block_size,
            'vocab_size': self.vocab_size,
            'emb_dim': self.emb_dim,
            'state_dict': self.state_dict()
        }

        torch.save(model_info, path)

    @staticmethod
    def load_model(path):
        """
        Load the model parameters.

        path: str: The path from where the model parameters should be loaded.
        """

        model_info = torch.load(path, map_location=torch.device('cpu'))
        model = MLP(block_size=model_info['block_size'], vocab_size=model_info['vocab_size'], emb_dim=model_info['emb_dim'])
        model.load_state_dict(model_info['state_dict'])
        return model