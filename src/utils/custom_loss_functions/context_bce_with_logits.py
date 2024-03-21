import torch.nn as nn


class ContextBNEWithLogits(nn.Module):
    """
    A custom PyTorch module that computes a weighted Binary Cross-Entropy (BCE) with logits loss, giving
    differential weighting to the last element of the input tensors.

    Args:
        query_loss_weight (float): The weight to be applied to the loss of the last element in the sequence, with
            the remaining elements receiving a computed weight such that the average weight across all elements
            equals 1.
        *args: Variable length argument list for BCEWithLogitsLoss.
        **kwargs: Arbitrary keyword arguments for BCEWithLogitsLoss.
    """
    def __init__(self, query_loss_weight, *args, **kwargs):
        super().__init__()
        self._query_loss_weight = query_loss_weight
        # Initialize the BCEWithLogitsLoss with no reduction to manually compute the weighted loss
        self._bne = nn.BCEWithLogitsLoss(*args, **kwargs, reduction='none')

    def forward(self, predictions, targets):
        """
        Computes the weighted BCEWithLogitsLoss, where the loss for the last element in each sequence
        is weighted by `query_loss_weight` and the rest of the elements are weighted such that the average
        weight across all elements is 1.

        Args:
            predictions (torch.Tensor): The predictions tensor, typically the output of a model,
                expected to have shape (batch_size, sequence_length).
            targets (torch.Tensor): The ground truth tensor, must have the same shape as the predictions tensor.

        Returns:
            torch.Tensor: The computed weighted loss as a scalar tensor.
        """
        # Compute element-wise losses
        losses = self._bne(predictions, targets)
        # Compute the mean loss for all elements except the last, and separately for the last element
        loss = (1 - self._query_loss_weight) * losses[:, :-1].mean() + self._query_loss_weight * losses[:, -1].mean()

        return loss
