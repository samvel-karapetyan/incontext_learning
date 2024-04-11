import torch

from src.baseline_methods.base_method import BaseMethod


class KNN(BaseMethod):
    """
    K-Nearest Neighbors (KNN) method for classification.

    Parameters:
        n_neighbors (int): Number of neighbors to consider (default is 1).
    """
    def __init__(self, n_neighbors=1):
        super(KNN, self).__init__(group_required=False)

        self._n_neighbors = n_neighbors

    def predict(self, x_train, y_train, x_test):
        """
        Predict labels for the test set using the KNN algorithm.

        Parameters:
            x_train (torch.Tensor): Training features.
            y_train (torch.Tensor): Training labels.
            x_test (torch.Tensor): Test features.

        Returns:
            torch.Tensor: Predicted labels for the test set.
        """
        # Compute distances between test point and all training points
        distances = torch.cdist(x_test, x_train)

        # Find indices of k nearest neighbors for each test point
        _, indices = distances.topk(self._n_neighbors, dim=1, largest=False)

        # Retrieve corresponding labels from training data
        neighbor_labels = y_train[indices]

        # Perform majority voting to determine the predicted label
        pred, _ = torch.mode(neighbor_labels, dim=1)
        return pred
