import torch
import torchmetrics


class GroupAccuracy(torchmetrics.Metric):
    """A metric that computes specific group accuracy.

    Attributes:
        group (int): Determines which group accuracy to compute.
        threshold (float): The threshold for binary classification, defaults to 0.5.
    """
    def __init__(self,
                 group: int,
                 threshold: float = 0.5):
        super().__init__()
        self.group = group
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
            self,
            query_prediction_batch: torch.Tensor,
            query_target_batch: torch.Tensor,
            query_spurious_batch: torch.Tensor,
    ):
        """
        Update the metric states based on context and query.

        Parameters:
        query_prediction_batch: A [0,1]-valued torch.Tensor of shape (B,) specifying predictions for queries.
            There is one query per in-context learning instance.
        query_target_batch: A {0,1}-valued torch.Tensor of shape (B,) specifying query class labels.
        query_spurious_batch: A {0,1}-valued torch.Tensor of shape (B,) specifying query spurious labels.
        """
        query_prediction_batch = (query_prediction_batch > self.threshold).int()
        for (query_prediction,
             query_target,
             query_spurious) in zip(query_prediction_batch,
                                    query_target_batch,
                                    query_spurious_batch):
            query_group = 2 * query_target + query_spurious

            if query_group == self.group:
                self.correct += (query_prediction == query_target)  # Increment correct if prediction matches target
                self.total += 1  # Increment total count

    def compute(self) -> float:
        """Computes the accuracy metric based on the current state."""
        return self.correct.float() / self.total
