import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from src.baseline_methods.base_method import BaseMethod


class ERM(BaseMethod):
    """
    Empirical Risk Minimization (ERM) method for binary classification.

    Parameters:
        n_epochs (int): Number of training epochs (default is 100).
        lr (float): Learning rate for the optimizer (default is 0.01).
    """
    def __init__(self, n_epochs=100, lr=0.01):
        super(ERM, self).__init__(group_required=False)

        self._n_epochs = n_epochs
        self._lr = lr

    def predict(self, x_train, y_train, x_test):
        """
        Train the ERM model and make predictions on the test set.

        Parameters:
            x_train (torch.Tensor): Training features.
            y_train (torch.Tensor): Training labels.
            x_test (torch.Tensor): Test features.

        Returns:
            torch.Tensor: Predicted labels for the test set.
        """
        model = nn.Linear(x_train.shape[1], 1)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self._lr)

        # Training loop
        for _ in range(self._n_epochs):  # Number of epochs
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        # Testing
        model.eval()
        with torch.no_grad():
            test_pred = model(x_test)
            test_pred_label = torch.sigmoid(test_pred).squeeze(0)  # Convert to binary label

        return test_pred_label
