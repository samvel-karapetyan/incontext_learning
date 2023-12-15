import torch
import torch.nn

from torchmetrics import Metric


class WorstGroupAccuracy(Metric):
    """
    A custom PyTorch Metric for calculating the accuracy of the worst-performing group. This metric is particularly
    useful in fairness-aware machine learning, where the goal is often to improve the model's performance on the
    underperforming groups.
    """

    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups
        self.add_state("correct", default=torch.zeros(num_groups), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_groups), dist_reduce_fx="sum")

    def update(self, preds, targets, groups):
        """
        Updates the state tensors with the new batch of predictions, targets, and group identifiers.

        Args:
            preds (torch.Tensor): The predictions tensor, typically the output of a model.
            targets (torch.Tensor): The ground truth tensor.
            groups (torch.Tensor): A tensor containing the group identifier for each instance.
        """
        preds = torch.argmax(preds, dim=-1)
        is_correct = (preds == targets).long()
        for g in range(self.num_groups):
            self.correct[g] += is_correct[groups == g].sum()
            self.total[g] += (groups == g).long().sum()

    def compute(self):
        """
        Computes the accuracy of the worst-performing group.

        Returns:
            torch.Tensor: The accuracy of the worst-performing group.
        """
        worst_group_acc = torch.tensor(1.0)
        for c, t in zip(self.correct, self.total):
            if t > 0:
                worst_group_acc = min(worst_group_acc, c.float() / t)
        return worst_group_acc

    def compute_all_groups(self):
        """
        Computes the accuracies of all groups.

        Returns:
            torch.Tensor: A tensor containing the accuracies for all groups, with 'nan' for groups with no instances.
        """
        ret = self.correct.float() / self.total
        ret[self.total == 0] = float('nan')
        return ret
