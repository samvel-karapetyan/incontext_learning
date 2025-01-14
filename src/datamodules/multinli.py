import logging
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader

from src.datamodules.datasets import MultiNLIForEncodingExtraction

log = logging.getLogger(__name__)


def custom_collate_fn(batch):
    x_values, indices = zip(*batch)
    return x_values, torch.tensor(indices)

class MultiNLIDataModule(pl.LightningDataModule):
    """
        A PyTorch Lightning data module for the MultiNLI dataset.

        This module is designed to handle the loading and preprocessing of the MultiNLI dataset for
        use in training, validation, and testing phases of a machine learning model. It leverages
        PyTorch's DataLoader for efficient data loading and the WILDS library's data loaders for
        handling training and evaluation datasets with a focus on generalization across distribution shifts.

        Attributes:
            name (str): Name of the dataset.
            root_dir (str): Root directory where the dataset is stored.
            resize_size (int): The size to which the smaller edge of the image will be resized.
            batch_size (int): The size of the batch for data loading.
            num_workers (int): The number of worker processes for data loading.
            with_subsets (bool): Indicates if the dataset is split into subsets (train, val, test).
    """
    def __init__(self, name: str, root_dir: str, batch_size: int, num_workers: int, **kwargs):
        super().__init__()
        self._name = name
        self._root_dir = root_dir
        self._batch_size = batch_size
        self._num_workers = num_workers

        self.with_subsets = True

        self._dataset = None

    def prepare_data(self):
        """Prepares the MultiNLI dataset for use, downloading it if necessary."""
        self._dataset = MultiNLIForEncodingExtraction(root_dir=self._root_dir, download=True)

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the training subset of the MultiNLI dataset."""
        train_data = self._dataset.get_subset("train")

        train_loader = DataLoader(train_data,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers,
                                collate_fn=custom_collate_fn)

        return train_loader

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the validation subset of the MultiNLI dataset."""
        val_data = self._dataset.get_subset("val")
        val_loader = DataLoader(val_data,
                                     batch_size=self._batch_size,
                                     num_workers=self._num_workers,
                                     collate_fn=custom_collate_fn)

        return val_loader

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the test subset of the MultiNLI dataset."""
        test_data = self._dataset.get_subset("test")
        test_loader = DataLoader(test_data,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers,
                                collate_fn=custom_collate_fn)

        return test_loader

    def get_combined_dataloader(self):
        sub_dataloaders = {"train": self.train_dataloader(),
                           "val": self.val_dataloader(),
                           "test": self.test_dataloader()}

        sub_dataloader_names = list(sub_dataloaders.keys())

        return CombinedLoader(sub_dataloaders, mode="sequential"), sub_dataloader_names

    @property
    def train_dataset(self):
        train_data = self._dataset.get_subset("train")

        return train_data

    @property
    def val_dataset(self):
        train_data = self._dataset.get_subset("val")

        return train_data

    @property
    def test_dataset(self):
        train_data = self._dataset.get_subset("test")

        return train_data
