import logging
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.datamodules.datasets import ImagenetEmbContextsDataset

log = logging.getLogger(__name__)


class ImagenetEmbContextsDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for the ImageNet Embedding Contexts dataset.

    This module handles the setup and provision of data loaders for the inner train,
    inner validation, and outer dataset splits.

    Attributes:
        dataset_path (str): Path to the dataset.
        encoding_extractor (str): Name of the encoding extractor.
        saved_val_sets_path (str or None): Path for loading validation sets; if None, new data is generated.
        data_length (int): Length of the dataset.
        inner_outer_val_len (int): Length of the inner-outer val set.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loaders.
        context_class_size (int): Size of each class in context.
        minority_group_proportion (float): Proportion of the minority group per class.
        are_spurious_tokens_fixed (bool): Flag indicating whether to use fixed spurious tokens.
        are_class_tokens_fixed (bool): Flag indicating whether to use fixed class tokens.
        token_generation_mode (str): Mode of token generation. Accepts 'random' or 'opposite'.
                                     'random' generates tokens with normal distribution,
                                     and 'opposite' generates a pair of tokens where the second is the negative of the first.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'separate_token'(x,c) , 'no_spurious'(x), 'sum'(x+c)
    """
    def __init__(self,
                 dataset_path,
                 encoding_extractor,
                 saved_path,
                 data_length,
                 batch_size,
                 num_workers,
                 context_class_size,
                 minority_group_proportion,
                 are_spurious_tokens_fixed,
                 are_class_tokens_fixed,
                 token_generation_mode,
                 spurious_setting,
                 *args, **kwargs):
        super(ImagenetEmbContextsDataModule, self).__init__()

        # Initializing dataset parameters
        self._dataset_path = dataset_path
        self._encoding_extractor = encoding_extractor
        self._saved_path = saved_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._context_class_size = context_class_size
        self._minority_group_proportion = minority_group_proportion
        self._are_spurious_tokens_fixed = are_spurious_tokens_fixed
        self._are_class_tokens_fixed = are_class_tokens_fixed
        self._token_generation_mode = token_generation_mode
        self._spurious_setting = spurious_setting

        self._data_length = data_length

        self._dataset = None

    def setup(self,  *args, **kwargs):
        """
        Initializes the dataset.
        """
        # Creating dataset instances for each split using class attributes
        self._dataset = ImagenetEmbContextsDataset(self._dataset_path,
                                                   self._encoding_extractor,
                                                   self._data_length,
                                                   self._context_class_size,
                                                   self._minority_group_proportion,
                                                   self._are_spurious_tokens_fixed,
                                                   self._are_class_tokens_fixed,
                                                   self._token_generation_mode,
                                                   self._spurious_setting,
                                                   self._saved_path)

    def val_dataloader(self):
        """
        Creates a DataLoader for the validation set.

        Returns:
            DataLoader: The DataLoader for the validation set.
        """
        return DataLoader(self._dataset, batch_size=self._batch_size, num_workers=self._num_workers)
