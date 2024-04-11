from abc import ABC, abstractmethod
from typing import Optional

import torch
import torchmetrics

from tqdm import tqdm

from pytorch_lightning.utilities import CombinedLoader
from src.utils.custom_metrics import GroupAccuracy, MinorityMajorityAccuracy, WorstGroupAccuracy
import pytorch_lightning as pl


class BaseMethod(ABC):
    """
    BaseMethod - Abstract Base Class for Method Validation

    This class provides a framework for validating machine learning methods. 
    It includes methods for validating the method using validation data and computing various metrics.
    """

    def validate(self, datamodule: pl.LightningDataModule):
        """
        Validates the method using the validation data provided by the datamodule.

        Args:
        datamodule (DataModule): A PyTorch Lightning datamodule containing the validation data.

        Returns:
        List[Dict[str, float]]: A list of dictionaries containing the evaluation results for each validation set.
        """
        # Get the validation dataloader from the datamodule
        dataloader = datamodule.val_dataloader()

        if isinstance(dataloader, CombinedLoader):
            # Create a dictionary of dataloaders for each split
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
            group_accuracies[set_name] = [GroupAccuracy(group=i) for i in range(4)]

        # Iterate over each split and its corresponding dataloader
        eval_results = []
        for set_name, loader in tqdm(dataloaders.items()):
            split_results = dict()
    
            # Iterate over each datapoint in the dataloader
            for datapoint in loader:
                # Extract input sequences, context, queries, and query indices from the datapoint
                input_seq, context, queries, query_indices = datapoint

                if input_seq.shape[0] != 1:
                    raise ValueError("Dataloader should have batch_size 1.")

                # for preparing context information we need to deduce spurious_setting from query_indices
                if query_indices[0] == 2:  # no_spurious or sum
                    context_x = input_seq[0, ::3]
                elif query_indices[0] == 3:  # separate_token or sum_with_spurious
                    context_x = input_seq[0, ::4]
                else:
                    raise ValueError("Could not decide spurious setting from query_indices: ", query_indices)

                context_c = context[0, :, 1]
                context_y = context[0, :, 2]

                # Prepare query information
                query_x = input_seq[0, [-1]]
                query_c = queries[0, -1, [1]]
                query_y = queries[0, -1, [2]]
                groups = 2 * context_y + context_c

                pred = self.predict(
                    x_train=context_x,
                    y_train=context_y,
                    x_test=query_x,
                    groups=groups)

                # Update accuracy metrics
                accuracy[set_name].update(pred, query_y)

                worst_group_accuracy[set_name].update(
                    preds=pred,
                    targets=query_y,
                    spurious_labels=query_c)

                for min_maj_metric in [accuracy_minority[set_name],
                                       accuracy_majority[set_name]]:
                    min_maj_metric.update(
                        query_prediction_batch=pred,
                        query_target_batch=query_y,
                        query_spurious_batch=query_c,
                        context_targets_batch=context_y.unsqueeze(0),
                        context_spurious_vals_batch=context_c.unsqueeze(0))

                for i in range(4):
                    group_accuracies[set_name][i].update(
                        query_prediction_batch=pred,
                        query_target_batch=query_y,
                        query_spurious_batch=query_c)

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
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_test: torch.Tensor,
            groups: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Should return a single prediction in [0, 1].

        Parameters:
        x_train: torch.Tensor of shape (num_examples, dim) containing training data.
        y_train: torch.Tensor of shape (num_examples,) containing binary labels.
        x_test: torch.Tensor of shape (1, dim) containing a single test example.
        groups: torch.Tensor of shape (num_examples,) containing groups of training examples (numbers in {0, 1, 2, 3}).
        """
        pass
