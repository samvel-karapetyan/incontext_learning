from abc import ABC, abstractmethod
from typing import Optional

import torch
import torchmetrics

from tqdm import tqdm

from pytorch_lightning.utilities import CombinedLoader
from src.utils.custom_metrics import GroupAccuracy, MinorityMajorityAccuracy, WorstGroupAccuracy


class BaseMethod(ABC):
    """
    BaseMethod - Abstract Base Class for Method Validation

    This class provides a framework for validating machine learning methods. 
    It includes methods for validating the method using validation data and computing various metrics.

    Attributes:
        group_required (bool): A boolean indicating whether the method requires group information.
    """
    def __init__(self, group_required):
        self._group_required = group_required

    def validate(self, datamodule):
        """
        Validates the method using the validation data provided by the datamodule.

        Args:
        datamodule (DataModule): A PyTorch Lightning datamodule containing the validation data.

        Returns:
        List[Dict[str, float]]: A list of dictionaries containing the evaluation results for each validation set.
        """
        # Validate the algorithm using the provided datamodule
        eval_results = []

        # Get the validation dataloader from the datamodule
        dataloader = datamodule.val_dataloader()

        # Check if the dataloader is a CombinedLoader
        if type(dataloader) == CombinedLoader:
            # If it is, create a dictionary of dataloaders for each split
            dataloaders = {split.name.lower(): loader for split, loader in dataloader.iterables.items()}
        else:
            # Otherwise, create a dictionary with a single "test" split
            dataloaders = {"test": dataloader}

        # Initialize dictionaries to store accuracy metrics
        accuracy = dict()
        worst_group_accuracy = dict()
        accuracy_minority = dict()
        accuracy_majority = dict()
        group_accuracies = dict()

        # Initialize accuracy metrics for each split and group
        for set_name in dataloaders.keys():
            accuracy[set_name] = torchmetrics.Accuracy(task="binary")
            worst_group_accuracy[set_name] = WorstGroupAccuracy()
            accuracy_minority[set_name] = MinorityMajorityAccuracy(group_type="minority")
            accuracy_majority[set_name] = MinorityMajorityAccuracy(group_type="majority")

            group_accuracies[set_name] = [dict() for _ in range(4)]
            for i in range(4):
                group_accuracies[set_name][i] = GroupAccuracy(group=i)

        # Iterate over each split and its corresponding dataloader
        for set_name, loader in tqdm(dataloaders.items()):
            split_results = dict()
    
            # Iterate over each datapoint in the dataloader
            for datapoint in loader:
                # Extract input sequences, context, queries, and query indices from the datapoint
                input_seq, context, queries, query_indices = datapoint

                # Extract context inputs
                context_x = input_seq[0, ::3]
                context_c = context[0, :, 1]
                context_y = context[0, :, 2]

                # Extract query inputs
                query_x = input_seq[0, [-1]]
                query_c = queries[0, -1, [1]]
                query_y = queries[0, -1, [2]]

                # Predict the target variable
                if self.group_required:
                    pred = self.predict(context_x, context_y, query_x, 2*context_y + context_c)
                else:
                    pred = self.predict(context_x, context_y, query_x)

                # Update accuracy metrics
                accuracy[set_name].update(pred, query_y) 
                worst_group_accuracy[set_name].update(pred, query_y, query_c)
                accuracy_minority[set_name].update(pred, query_y, query_c, context_y.unsqueeze(0), context_c.unsqueeze(0))
                accuracy_majority[set_name].update(pred, query_y, query_c, context_y.unsqueeze(0), context_c.unsqueeze(0))
                for i in range(4):
                    group_accuracies[set_name][i].update(pred, query_y, query_c)

            # Compute and store evaluation results for the split
            split_results[f"val_{set_name}_accuracy"] = accuracy[set_name].compute()
            split_results[f"val_{set_name}_worst_group_accuracy"] = worst_group_accuracy[set_name].compute()
            split_results[f"val_{set_name}_minority_accuracy"] = accuracy_minority[set_name].compute()
            split_results[f"val_{set_name}_majority_accuracy"] = accuracy_majority[set_name].compute()
            for i in range(4):
                split_results[f"val_{set_name}_group_{i}_accuracy"] = group_accuracies[set_name][i].compute()

            # Append evaluation results for the split to the overall evaluation results
            eval_results.append(split_results)

        return eval_results
    
    @abstractmethod
    def predict(
        x_train: torch.Tensor, 
        y_train: torch.Tensor, 
        x_test: torch.Tensor,
        groups: Optional[torch.Tensor],
        *args, **kwargs
    ) -> int:
        pass

    @property
    def group_required(self):
        # Property that returns whether the method requires group information
        return self._group_required
