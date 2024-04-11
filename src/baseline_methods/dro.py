import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from src.baseline_methods.base_method import BaseMethod


class DRO(BaseMethod):
    """
    Distributionally Robust Optimization (DRO) method for binary classification.

    Parameters:
        n_epochs (int): Number of training epochs (default is 100).
        lr (float): Learning rate for the optimizer (default is 0.01).
        group_weight_step (float): Step size for updating group weights (default is 0.01).
    """
    def __init__(self, n_epochs=100, lr=0.01, group_weight_step=0.01):
        super(DRO, self).__init__(group_required=True)

        self._n_epochs = n_epochs
        self._lr = lr
        self._group_weight_step = group_weight_step

    def predict(self, x_train, y_train, x_test, groups):
        """
        Train the DRO model and make predictions on the test set.

        Parameters:
            x_train (torch.Tensor): Training features.
            y_train (torch.Tensor): Training labels.
            x_test (torch.Tensor): Test features.
            groups (torch.Tensor): Group labels for each sample.

        Returns:
            torch.Tensor: Predicted labels for the test set.
        """
        n_groups = groups.max().item() + 1

        # Define the linear model and move it to the device
        model = nn.Linear(x_train.shape[1], 1)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)

        # Initialize group weights and move them to the device
        group_weights = torch.ones(n_groups, dtype=torch.float32) / n_groups

        # Training loop
        for _ in range(self._n_epochs):  # Number of epochs
            model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_train)
            losses = criterion(outputs, y_train.unsqueeze(1).float()).squeeze(1)

            # Calculate loss per group
            group_losses = torch.zeros(n_groups)
            for group in range(n_groups):
                group_mask = (groups == group)
                if group_mask.any():
                    group_losses[group] = (losses * group_mask.float()).sum() / group_mask.float().sum()

            # Update group weights
            group_weights = group_weights * torch.exp(self._group_weight_step * group_losses)
            group_weights = group_weights / group_weights.sum()
            group_weights = group_weights.detach()

            # Weighted loss for optimization
            weighted_loss = (group_losses * group_weights).sum()
            weighted_loss.backward()
            optimizer.step()

        # Testing
        model.eval()
        with torch.no_grad():
            test_pred = model(x_test)
            test_pred_label = torch.sigmoid(test_pred).squeeze(0)  # Convert to binary label

        return test_pred_label
