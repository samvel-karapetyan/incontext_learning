from collections import Counter

import torch
import torchmetrics


class MinorityMajorityAccuracy(torchmetrics.Metric):
    """A metric that can compute minority or majority group accuracy.

    Attributes:
    group_type (str): Determines whether to focus on the 'majority' or 'minority' group.
    threshold (float): The threshold for binary classification, defaults to 0.5.
    """
    def __init__(self,
                 group_type: str,
                 threshold: int = 0.5):
        super().__init__()
        self.group_type = group_type
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
            self,
            query_prediction_batch: torch.Tensor,
            query_target_batch: torch.Tensor,
            query_spurious_batch: torch.Tensor,
            context_targets_batch: torch.Tensor,
            context_spurious_vals_batch: torch.Tensor,
    ):
        """
        Update the metric states based on context and query.

        Parameters:
        query_prediction_batch: A [0,1]-valued torch.Tensor of shape (B,) specifying predictions for queries.
            There is one query per in-context learning instance.
        query_target_batch: A {0,1}-valued torch.Tensor of shape (B,) specifying query class labels.
        query_spurious_batch: A {0,1}-valued torch.Tensor of shape (B,) specifying query spurious labels.
        context_targets_batch: A {0,1}-valued torch.Tensor of shape (B, num_icl_examples) specifying class
            labels of context examples.
        context_spurious_vals_batch: A {0,1}-valued torch.Tensor of shape (B, num_icl_examples) specifying spurious
            labels of context examples.
        """
        query_prediction_batch = (query_prediction_batch > self.threshold).int()
        for (query_prediction,
             query_target,
             query_spurious,
             context_targets,
             context_spurious_vals) in zip(query_prediction_batch,
                                           query_target_batch,
                                           query_spurious_batch,
                                           context_targets_batch,
                                           context_spurious_vals_batch):
            query_group = 2 * query_target + query_spurious
            context_groups = 2 * context_targets + context_spurious_vals
            selected_groups = self.select_groups(context_groups)

            if query_group in selected_groups:
                self.correct += (query_prediction == query_target)  # Increment correct if prediction matches target
                self.total += 1  # Increment total count

    def compute(self) -> float:
        """Computes the accuracy metric based on the current state."""
        return self.correct.float() / self.total

    def select_groups(
            self,
            context_groups: torch.Tensor,
    ) -> list[int]:
        """Returns list of groups for which accuracy is tracked."""
        counts = Counter({k: 0 for k in range(4)})
        counts.update(context_groups.tolist())  # Count the occurrences of each group

        if self.group_type == 'majority':
            target_occurrence = max(counts.values())  # Maximum occurrence for majority
        elif self.group_type == 'minority':
            target_occurrence = min(counts.values())  # Minimum occurrence for minority
        else:
            raise ValueError("Invalid group type. Choose 'majority' or 'minority'.")

        selected_groups = [num for num, count in counts.items() if count == target_occurrence]
        return selected_groups
