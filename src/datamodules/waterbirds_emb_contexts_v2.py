import logging

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from src.datamodules.datasets import WaterbirdsEmbContextsDatasetV2

log = logging.getLogger(__name__)


class WaterbirdsEmbContextsDataModuleV2(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for the Waterbirds Embedding Contexts dataset.

    Currently suited only for evaluations. TODO(hrayr): adapt for training.
    """

    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 data_length: int,
                 saved_val_set_path,
                 batch_size,
                 num_workers,
                 context_class_size: int,
                 group_proportions: list[float],
                 spurious_setting: str,
                 v1_behavior: bool,
                 *args, **kwargs):
        """
        Args:

        root_dir (str): The root directory of the dataset.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        saved_val_set_path (str or None): Path for loading validation set; if None, new data is generated.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loaders.
        context_class_size (int): The size of each class in the context.
        group_proportions (list[float]): Proportions for the 4 groups.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'no_spurious'(x), 'sum'(x+c), 'separate_token'(x, c)
                                or 'sum_with_spurious'(x+c, c).
        v1_behavior (bool): Whether intermediate queries should be the context examples.
        """
        super(WaterbirdsEmbContextsDataModuleV2, self).__init__()

        # Initializing dataset parameters
        self._root_dir = root_dir
        self._encoding_extractor = encoding_extractor
        self._data_length = data_length
        self._saved_val_set_path = saved_val_set_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._context_class_size = context_class_size
        self._group_proportions = group_proportions
        self._spurious_setting = spurious_setting
        self._v1_behavior = v1_behavior

        self._dataset = None

    def setup(self, *args, **kwargs):
        """
        Initializes the dataset.
        """
        # Creating dataset instances for each split using class attributes
        self._dataset = WaterbirdsEmbContextsDatasetV2(
            root_dir=self._root_dir,
            encoding_extractor=self._encoding_extractor,
            data_length=self._data_length,
            context_class_size=self._context_class_size,
            group_proportions=self._group_proportions,
            spurious_setting=self._spurious_setting,
            v1_behavior=self._v1_behavior,
            saved_data_path=self._saved_val_set_path)

    def val_dataloader(self):
        return DataLoader(self._dataset, batch_size=self._batch_size, num_workers=self._num_workers)
