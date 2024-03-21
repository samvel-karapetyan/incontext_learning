import torch
import torchmetrics


class GroupAccuracy(torchmetrics.Metric):
    """
    A custom metric class in PyTorch to calculate accuracy for the specified group.

    Attributes:
    threshold (float): The threshold for binary classification, defaults to 0.5.
    correct (torch.Tensor): Tracks the number of correct predictions.
    total (torch.Tensor): Tracks the total number of observations.

    Methods:
    update(predictions, targets, spurious_labels): Updates the state with new predictions and targets.
    compute(): Computes the final accuracy metric.
    """

    def __init__(self, group, threshold=0.5):
        super().__init__()
        self.group = group
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, spurious_labels: torch.Tensor):
        """
        Update the metric states based on the provided batch of predictions and targets.

        Parameters:
        predictions (torch.Tensor): Model predictions for the current batch.
        targets (torch.Tensor): Ground truth labels for the current batch.
        spurious_labels (torch.Tensor): Additional labels used to define groups for analysis.
        """
        predictions = (predictions > self.threshold).int()
        for prediction, target, spurious_label in zip(predictions, targets, spurious_labels):
            group = 2 * target + spurious_label  # Form a unique identifier for group

            if group[-1] == self.group:
                self.correct += (prediction[-1] == target[-1])  # Increment correct if prediction matches target
                self.total += 1  # Increment total count

    def compute(self):
        """
        Compute the accuracy metric based on the current state.

        Returns:
        float: The computed accuracy value.
        """
        return self.correct.float() / self.total
