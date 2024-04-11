import torch

from src.baseline_methods.base_method import BaseMethod


class KNN(BaseMethod):
    """K-Nearest Neighbors (KNN) method for classification."""
    def __init__(self, n_neighbors: int = 1):
        super(KNN, self).__init__()
        self._n_neighbors = n_neighbors

    def predict(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_test: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        # Compute distances between test point and all training points
        distances = torch.cdist(x_test, x_train)

        # Find indices of k nearest neighbors for each test point
        _, indices = distances.topk(self._n_neighbors, dim=1, largest=False)

        # Retrieve corresponding labels from training data
        neighbor_labels = y_train[indices]

        # Perform majority voting to determine the predicted label
        pred, _ = torch.mode(neighbor_labels, dim=1)
        return pred
