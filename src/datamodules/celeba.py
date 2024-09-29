import logging
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from torchvision.transforms import transforms
from pytorch_lightning.utilities import CombinedLoader

from src.datamodules.datasets import CelebAForEncodingExtraction

log = logging.getLogger(__name__)


class CelebADataModule(pl.LightningDataModule):
    """
        A PyTorch Lightning data module for the CelebA dataset.

        This module is designed to handle the loading and preprocessing of the CelebA dataset for
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
    def __init__(self, name: str, root_dir: str, resize_size: int, batch_size: int, num_workers: int, **kwargs):
        super().__init__()
        self._name = name
        self._root_dir = root_dir
        self._batch_size = batch_size
        self._num_workers = num_workers

        self.with_subsets = True

        # https://github.com/kohpangwei/group_DRO/blob/cbbc1c5b06844e46b87e264326b56056d2a437d1/data/celebA_dataset.py#L81
        self._transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._dataset = None

    def prepare_data(self):
        """Prepares the CelebA dataset for use, downloading it if necessary."""
        self._dataset = CelebAForEncodingExtraction(root_dir=self._root_dir, download=True)

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the training subset of the CelebA dataset."""
        train_data = self._dataset.get_subset("train", transform=self._transform)

        train_loader = get_train_loader("standard", train_data,
                                        batch_size=self._batch_size,
                                        num_workers=self._num_workers)

        return train_loader

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the validation subset of the CelebA dataset."""
        val_data = self._dataset.get_subset("val", transform=self._transform)
        val_loader = get_eval_loader("standard", val_data,
                                     batch_size=self._batch_size,
                                     num_workers=self._num_workers)

        return val_loader

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the test subset of the CelebA dataset."""
        test_data = self._dataset.get_subset("test", transform=self._transform)
        test_loader = get_eval_loader("standard", test_data,
                                      batch_size=self._batch_size,
                                      num_workers=self._num_workers)

        return test_loader

    def get_combined_dataloader(self):
        sub_dataloaders = {"train": self.train_dataloader(),
                           "val": self.val_dataloader(),
                           "test": self.test_dataloader()}

        sub_dataloader_names = list(sub_dataloaders.keys())

        return CombinedLoader(sub_dataloaders, mode="sequential"), sub_dataloader_names

    @property
    def train_dataset(self):
        train_data = self._dataset.get_subset("train", transform=self._transform)

        return train_data

    @property
    def val_dataset(self):
        train_data = self._dataset.get_subset("val", transform=self._transform)

        return train_data

    @property
    def test_dataset(self):
        train_data = self._dataset.get_subset("test", transform=self._transform)

        return train_data
