import pytorch_lightning as pl

from torch.utils.data import DataLoader

from src.datamodules.datasets import ImagenetDataset
from pytorch_lightning.utilities import CombinedLoader


class ImagenetDataModule(pl.LightningDataModule):
    """
    A data module for handling the ImageNet dataset.

    This class facilitates the creation of a DataLoader for the ImageNet dataset, allowing for easy
    batching, shuffling, and parallel loading of the data using PyTorch. It encapsulates dataset initialization,
    including resizing and cropping of images, and provides a method to obtain a DataLoader for the dataset.

    Args:
        dataset_path (str): The file path to the dataset directory.
        resize_size (int): The size to which the smaller edge of the image will be resized.
        crop_size (int): The size of the square crop to be extracted from the center of the image.
        batch_size (int): The number of items to be included in each batch.
        num_workers (int): The number of subprocesses to use for data loading.
    """

    def __init__(self,
                 dataset_path,
                 resize_size,
                 crop_size,
                 batch_size,
                 num_workers,
                 *args, **kwargs):
        super(ImagenetDataModule, self).__init__()
        self.dataset_path = dataset_path
        self.resize_size = resize_size
        self.crop_size = crop_size

        self.with_subsets = True

        # Store batch size and number of workers for data loading
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._train_set = None
        self._val_set = None
        self._test_set = None

    def prepare_data(self):
        """
        Prepares the ImageNet dataset for use with the DataLoader.

        This method initializes the ImageNet with the specified configuration, including resizing and
        cropping of images.
        """
        self._train_set = ImagenetDataset(self.dataset_path, self.resize_size, self.crop_size, split="train")
        self._val_set = ImagenetDataset(self.dataset_path, self.resize_size, self.crop_size, split="val")
        self._test_set = ImagenetDataset(self.dataset_path, self.resize_size, self.crop_size, split="test")

    def train_dataloader(self):
        """
        Creates and returns a train DataLoader for the ImageNet dataset.

        Returns:
            DataLoader: A DataLoader instance for the ImageNet dataset.
        """
        # Initialize the DataLoader with the dataset, batch size, and number of workers
        dataloader = DataLoader(self._train_set,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers)

        return dataloader

    def val_dataloader(self):
        """
        Creates and returns a val DataLoader for the ImageNet dataset.

        Returns:
            DataLoader: A DataLoader instance for the ImageNet dataset.
        """
        # Initialize the DataLoader with the dataset, batch size, and number of workers
        dataloader = DataLoader(self._val_set,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers)

        return dataloader

    def test_dataloader(self):
        """
        Creates and returns a test DataLoader for the ImageNet dataset.

        Returns:
            DataLoader: A DataLoader instance for the ImageNet dataset.
        """
        # Initialize the DataLoader with the dataset, batch size, and number of workers
        dataloader = DataLoader(self._test_set,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers)

        return dataloader

    def get_combined_dataloader(self):
        sub_dataloaders = {"train": self.train_dataloader(),
                           "val": self.val_dataloader(),
                           "test": self.test_dataloader()}

        sub_dataloader_names = list(sub_dataloaders.keys())

        return CombinedLoader(sub_dataloaders, mode="sequential"), sub_dataloader_names
