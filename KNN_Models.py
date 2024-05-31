import os
import torch
import numpy as np

# Set env CUDA_LAUNCH_BLOCKING=1
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KMeans:
    """
    A KMeans clustering algorithm.
    """
    def __init__(self, n_clusters: int = 3):
        """
        Constructor for KMeans.

        n_clusters: int: The number of clusters.
        """
        self.n_clusters = n_clusters
        self.centroids = []

    def fit(self, X : torch.Tensor, max_iter : int = 100, tol : float = 1e-4):
        """
        X: torch.Tensor: The data to cluster.
        max_iter: int: The maximum number of iterations.
        tol: float: The tolerance for convergence.
        """

        X = X.to(device)
        self.findCenters(X)
        
        for _ in range(max_iter):
            distances = torch.cdist(X, torch.stack(self.centroids))
            labels = torch.argmin(distances, dim=1)

            centroids = []
            for i in range(self.n_clusters):
                if (labels == i).any():
                    centroids.append(X[labels == i].mean(dim=0))
                else:
                    centroids.append(X[np.random.randint(0, X.shape[0])])

            if torch.allclose(torch.stack(centroids), torch.stack(self.centroids), atol=tol):
                break
            
            self.centroids = centroids

    def predict(self, X):
        """
        X: torch.Tensor: The data to cluster.

        Returns: torch.Tensor: The cluster labels.
        """

        X = X.to(device)
        distances = torch.cdist(X, torch.stack(self.centroids))
        return torch.argmin(distances, dim=1)

    def findCenters(self, X):
        """
        X: torch.Tensor: The data to cluster.
        """

        self.centroids = [X[np.random.randint(0, X.shape[0])]]
        
        for _ in range(self.n_clusters - 1):
            distances = torch.sum(torch.stack([torch.norm(X - c, dim=1) for c in self.centroids]), dim=0)
            self.centroids.append(X[torch.argmax(distances)])