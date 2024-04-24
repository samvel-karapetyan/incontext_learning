from typing import Optional
import torch
import torch.nn as nn

from src.baseline_methods.base_method import BaseMethod


class ReweightedERM(BaseMethod):
    """Reweighted Empirical Risk Minimization (ERM) method for binary classification."""
    def __init__(self,
                 n_epochs: int = 100,
                 lr: float = 0.01,
                 device: str = "cpu"):
        super(ReweightedERM, self).__init__()
        self._n_epochs = n_epochs
        self._lr = lr
        self._device = device

    def predict(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_test: torch.Tensor,
            groups: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert groups is not None
        x_train, y_train, x_test = self.tensors_to_device(x_train, y_train, x_test, device=self._device)

        model = nn.Linear(x_train.shape[1], 1, device=self._device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(reduction='none')  # Use reduction='none' to get individual losses
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)

        # compute sample weights based on group counts
        group_counts = torch.zeros(4, dtype=torch.float32)
        for g_idx in range(4):
            group_counts[g_idx] = torch.sum(groups == g_idx)
        sample_weights = 1.0 / group_counts[groups]
        sample_weights = sample_weights.to(self._device)
        sample_weights /= sample_weights.sum()

        # Training loop
        for _ in range(self._n_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train.unsqueeze(1).float())
            loss = torch.sum(loss * sample_weights)  # Weighted loss
            loss.backward()
            optimizer.step()

        # Testing
        model.eval()
        with torch.no_grad():
            test_pred = model(x_test).detach().cpu()
            test_pred_label = torch.sigmoid(test_pred).squeeze(0)  # Convert to probability

        return test_pred_label
