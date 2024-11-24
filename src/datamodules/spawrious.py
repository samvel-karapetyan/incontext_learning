from torch.utils.data import DataLoader
from src.datamodules.datasets import SpawriousForEncodingExtraction


class SpawriousDataModule:
    """A data module for handling the our custom Spawrious dataset."""

    def __init__(self,
                 dataset_path: str,
                 resize_size: int,
                 batch_size: int,
                 num_workers: int,
                 *args, **kwargs):
        self.dataset_path = dataset_path
        self.resize_size = resize_size
        self.with_subsets = False

        # Store batch size and number of workers for data loading
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._dataset = None

    def prepare_data(self):
        self._dataset = SpawriousForEncodingExtraction(
            dataset_path=self.dataset_path,
            resize_size=self.resize_size)

    def get_dataloader(self):
        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers
        )
