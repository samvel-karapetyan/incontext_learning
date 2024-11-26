import torch

from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLRWithWarmup(_LRScheduler):
    """
    A PyTorch learning rate scheduler that combines cosine annealing with a warm-up phase.

    During the warm-up phase, the learning rate increases linearly from a minimum learning rate to the
    optimizer's initial learning rate over a specified number of steps. After the warm-up phase, the
    scheduler switches to cosine annealing, gradually decreasing the learning rate from the optimizer's
    initial learning rate to a minimum learning rate over the remaining steps.

    The total_steps parameter can be an integer or a string in the format "x/y", where x and y are integers.
    In the case of a string, total_steps is calculated as the integer division of x by y.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for the warm-up phase.
        total_steps (int or str): Total number of steps for training, including both warm-up and annealing phases.
                                  Can be an integer or a string in "x/y" format.
        min_lr (float, optional): Minimum learning rate after annealing. Default: 0.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps

        if isinstance(total_steps, str):
            self.total_steps = int(eval(total_steps))
        else:
            self.total_steps = total_steps

        self.min_lr = min_lr
        self.max_lr = optimizer.defaults['lr']
        super(CosineAnnealingLRWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = (self.max_lr - self.min_lr) * self.last_epoch / self.warmup_steps + self.min_lr
        else:
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + torch.cos(
                torch.tensor(self.last_epoch - self.warmup_steps) / (
                            self.total_steps - self.warmup_steps) * torch.pi))
        return [lr for _ in self.optimizer.param_groups]