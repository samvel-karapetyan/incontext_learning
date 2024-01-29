import logging
import os.path

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader

from src.datamodules.datasets import WaterbirdsEmbContextsDataset

log = logging.getLogger(__name__)


class WaterbirdsEmbContextsDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for the iNaturalist Embedding Contexts dataset.

    This module handles the setup and provision of data loaders for the inner train,
    inner validation, and outer dataset splits.

    Attributes:
        root_dir (str): Path to the dataset.
        encoding_extractor (str): Name of the encoding extractor.
        data_length (int):
        saved_val_set_path (str or None): Path for loading validation set; if None, new data is generated.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loaders.
        context_class_size (int): Size of each class in context.
        group_proportions:
        are_spurious_tokens_fixed (bool): Flag indicating whether to use fixed spurious tokens.
        are_class_tokens_fixed (bool): Flag indicating whether to use fixed class tokens.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'separate_token'(x,c) , 'no_spurious'(x), 'sum'(x+c)
        token_generation_mode (str): Mode of token generation. Accepts 'random' or 'opposite'.
                                     'random' generates tokens with normal distribution,
                                     and 'opposite' generates a pair of tokens where the second is the negative of the first.
    """

    def __init__(self,
                 root_dir,
                 encoding_extractor,
                 data_length,
                 saved_val_set_path,
                 batch_size,
                 num_workers,
                 context_class_size,
                 group_proportions,
                 are_spurious_tokens_fixed,
                 are_class_tokens_fixed,
                 spurious_setting,
                 token_generation_mode,
                 *args, **kwargs):
        super(WaterbirdsEmbContextsDataModule, self).__init__()

        # Initializing dataset parameters
        self._root_dir = root_dir
        self._encoding_extractor = encoding_extractor
        self._data_length = data_length
        self._saved_val_set_path = saved_val_set_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._context_class_size = context_class_size
        self._group_proportions = group_proportions
        self._are_spurious_tokens_fixed = are_spurious_tokens_fixed
        self._are_class_tokens_fixed = are_class_tokens_fixed
        self._token_generation_mode = token_generation_mode
        self._spurious_setting = spurious_setting

        self._dataset = None

    def setup(self, *args, **kwargs):
        """
        Initializes the dataset.
        """
        # Creating dataset instances for each split using class attributes
        self._dataset = WaterbirdsEmbContextsDataset(self._root_dir,
                                                     self._encoding_extractor,
                                                     self._data_length,
                                                     self._context_class_size,
                                                     self._group_proportions,
                                                     self._are_spurious_tokens_fixed,
                                                     self._are_class_tokens_fixed,
                                                     self._token_generation_mode,
                                                     self._spurious_setting,
                                                     self._saved_val_set_path)

    def val_dataloader(self):
        """
        Creates a DataLoader for the validation set.

        Returns:
            DataLoader: The DataLoader for the validation set.
        """
        return DataLoader(self._dataset, batch_size=self._batch_size, num_workers=self._num_workers)
