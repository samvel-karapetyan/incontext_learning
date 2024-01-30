import torch
import torchmetrics
from collections import Counter


class MinorityMajorityAccuracy(torchmetrics.Metric):
    """
    A custom metric class in PyTorch to calculate accuracy focusing on either the majority or minority groups
    in a dataset. This metric is useful for evaluating model performance across different demographic groups,
    ensuring fair representation.

    Attributes:
    group_type (str): Determines whether to focus on the 'majority' or 'minority' group.
    threshold (float): The threshold for binary classification, defaults to 0.5.
    correct (torch.Tensor): Tracks the number of correct predictions.
    total (torch.Tensor): Tracks the total number of observations.

    Methods:
    update(predictions, targets, spurious_labels): Updates the state with new predictions and targets.
    compute(): Computes the final accuracy metric.
    select_groups(tensor, group_type): Static method to select either majority or minority groups.
    """

    def __init__(self, group_type, threshold=0.5):
        super().__init__()
        self.group_type = group_type
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
            selected_groups = self.select_groups(group[:-1], group_type=self.group_type)

            if group[-1] in selected_groups:
                self.correct += (prediction[-1] == target[-1])  # Increment correct if prediction matches target
                self.total += 1  # Increment total count

    def compute(self):
        """
        Compute the accuracy metric based on the current state.

        Returns:
        float: The computed accuracy value.
        """
        return self.correct.float() / self.total

    @staticmethod
    def select_groups(tensor, group_type):
        """
        Selects groups from the tensor based on the frequency of occurrence, either as a majority or minority.

        Parameters:
        tensor (torch.Tensor): A tensor of group indices.
        group_type (str): A string that can be either 'majority' or 'minority' to select the group accordingly.

        Returns:
        list: A list containing the selected group indices based on the group type.
        """
        counts = Counter({k: 0 for k in range(4)})
        counts.update(tensor.tolist())  # Count the occurrences of each group

        if group_type == 'majority':
            target_occurrence = max(counts.values())  # Maximum occurrence for majority
        elif group_type == 'minority':
            target_occurrence = min(counts.values())  # Minimum occurrence for minority
        else:
            raise ValueError("Invalid group type. Choose 'majority' or 'minority'.")

        selected_groups = [num for num, count in counts.items() if count == target_occurrence]
        return selected_groups
