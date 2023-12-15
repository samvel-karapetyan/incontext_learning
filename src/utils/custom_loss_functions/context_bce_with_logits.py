import torch.nn as nn


class ContextBNEWithLogits(nn.Module):
    """
    A custom PyTorch loss module that applies Binary Cross-Entropy (BCE) with logits loss to the last 'n' elements
    of the predictions and targets tensors. This is useful for cases where only a subset of the context (the last 'n' elements)
    in a sequence is relevant for the loss calculation.
    """
    def __init__(self, keep_n, *args, **kwargs):
        super().__init__()
        self._keep_n = keep_n
        self._bne = nn.BCEWithLogitsLoss(*args, **kwargs)

    def forward(self, predictions, targets):
        """
        Applies the BCEWithLogitsLoss to the last 'keep_n' elements of the predictions and targets tensors.

        Args:
            predictions (torch.Tensor): The predictions tensor, typically the output of a model.
            targets (torch.Tensor): The ground truth tensor.

        Returns:
            torch.Tensor: The computed loss value.
        """
        predictions = predictions[:, -self._keep_n:]
        targets = targets[:, -self._keep_n:]

        return self._bne(predictions, targets)
