from torch.utils.data import DataLoader
from src.datamodules.datasets import CUBDataset


class CUBDataModule:
    """
    A data module for handling the Caltech-UCSD Birds (CUB) 200-2011 dataset.

    This class facilitates the creation of a DataLoader for the CUB-200-2011 dataset, allowing for easy
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
        self.dataset_path = dataset_path
        self.resize_size = resize_size
        self.crop_size = crop_size

        self.with_subsets = False

        # Store batch size and number of workers for data loading
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._dataset = None

    def prepare_data(self):
        """
        Prepares the CUB-200-2011 dataset for use with the DataLoader.

        This method initializes the CUB-200-2011 with the specified configuration, including resizing and
        cropping of images.
        """
        self._dataset = CUBDataset(self.dataset_path, self.resize_size, self.crop_size)

    def get_dataloader(self):
        """
        Creates and returns a DataLoader for the CUB-200-2011 dataset.

        Returns:
            DataLoader: A DataLoader instance for the ImageNet dataset.
        """
        # Initialize the DataLoader with the dataset, batch size, and number of workers
        dataloader = DataLoader(self._dataset,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers)

        return dataloader
