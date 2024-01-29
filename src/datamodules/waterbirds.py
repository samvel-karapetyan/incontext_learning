import logging
import pytorch_lightning as pl

from typing import NoReturn
from torch.utils.data import DataLoader
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from torchvision.transforms import transforms
from pytorch_lightning.utilities import CombinedLoader

from src.datamodules.datasets import WaterbirdsDataset

log = logging.getLogger(__name__)


class WaterbirdsDataModule(pl.LightningDataModule):
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

    def prepare_data(self) -> NoReturn:
        self._dataset = WaterbirdsDataset(root_dir=self._root_dir, download=True)

    def train_dataloader(self) -> DataLoader:
        train_data = self._dataset.get_subset("train", transform=self._transform)

        train_loader = get_train_loader("standard", train_data,
                                        batch_size=self._batch_size,
                                        num_workers=self._num_workers)

        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_data = self._dataset.get_subset("val", transform=self._transform)
        val_loader = get_eval_loader("standard", val_data,
                                     batch_size=self._batch_size,
                                     num_workers=self._num_workers)

        return val_loader

    def test_dataloader(self) -> DataLoader:
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
