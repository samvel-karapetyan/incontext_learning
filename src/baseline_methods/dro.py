from typing import Optional

import torch
import torch.nn as nn

from src.baseline_methods.base_method import BaseMethod


class DRO(BaseMethod):
    """Distributionally Robust Optimization (DRO) method for binary classification."""
    def __init__(self,
                 n_epochs: int = 100,
                 lr: float = 0.01,
                 group_weight_step: float = 0.01, 
                 device: str = "cpu"):
        super(DRO, self).__init__()
        self._n_epochs = n_epochs
        self._lr = lr
        self._group_weight_step = group_weight_step
        self._device = device

    def predict(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_test: torch.Tensor,
            groups: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert groups is not None
        x_train, y_train, x_test,  groups = self.tensors_to_device(
                                                x_train, 
                                                y_train, 
                                                x_test, 
                                                groups,
                                                device=self._device
                                            )
        
        n_groups = groups.max().item() + 1

        # Define the linear model
        model = nn.Linear(x_train.shape[1], 1, device=self._device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)

        # Initialize group weights
        group_weights = torch.ones(n_groups, dtype=torch.float32, device=self._device) / n_groups

        # Training loop
        for _ in range(self._n_epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_train)
            losses = criterion(outputs, y_train.unsqueeze(1).float()).squeeze(1)

            # Calculate loss per group
            group_losses = torch.zeros(n_groups, device=self._device)
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
            test_pred = model(x_test).detach().cpu()
            test_pred_label = torch.sigmoid(test_pred).squeeze(0)  # Convert to probability

        return test_pred_label
