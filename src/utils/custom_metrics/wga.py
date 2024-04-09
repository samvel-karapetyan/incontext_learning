import torch
import torchmetrics


class WorstGroupAccuracy(torchmetrics.Metric):

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct", default=torch.zeros(4), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(4), dist_reduce_fx="sum")

    def update(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            spurious_labels: torch.Tensor,
    ):
        """Updates the state tensors with the new batch of predictions, targets, and spurious labels.

        Args:
            preds: A [0,1]-valued torch.Tensor of shape (B,).
            targets: A {0,1}-valued torch.Tensor of shape (B,).
            spurious_labels: A {0,1}-valued torch.Tensor of shape (B,).
        """
        preds = (preds > self.threshold).int()
        is_correct = (preds == targets).long()
        groups = 2 * targets + spurious_labels
        for g in range(4):
            self.correct[g] += is_correct[groups == g].sum()
            self.total[g] += (groups == g).long().sum()

    def compute(self) -> float:
        worst_group_acc = torch.tensor(1.0)
        for c, t in zip(self.correct, self.total):
            if t > 0:
                worst_group_acc = min(worst_group_acc, c.float() / t)
        return worst_group_acc
