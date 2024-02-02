import logging
import os.path

import pytorch_lightning as pl

from enum import Enum
from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader

from src.datamodules.datasets import INaturalistEmbContextsDataset

log = logging.getLogger(__name__)


class INaturalistEmbContextsDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for the iNaturalist Embedding Contexts dataset.

    This module handles the setup and provision of data loaders for the inner train,
    inner validation, and outer dataset splits.

    Attributes:
        dataset_path (str): Path to the dataset.
        encoding_extractor (str): Name of the encoding extractor.
        saved_val_sets_path (str or None): Path for loading validation sets; if None, new data is generated.
        inner_train_len (int): Length of the inner training set.
        inner_val_len (int): Length of the inner validation set.
        outer_val_len (int): Length of the outer val set.
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
        rotate_encodings (bool): Determines if image encodings are rotated in training set. True enables rotation
                                 based on class labels, while False bypasses rotation.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'separate_token'(x,c) , 'no_spurious'(x), 'sum'(x+c)
    """
    class ValSets(Enum):
        """
        Enumeration for validation splits used in INaturalistEmbContextsDataModule.

        This enum defines constants for different types of validation splits
        such as inner, outer, and a combined inner-outer validation set.
        """
        INNER = "inner"
        OUTER = "outer"
        INNER_OUTER = "inner_outer"

    def __init__(self,
                 dataset_path,
                 encoding_extractor,
                 saved_val_sets_path,
                 inner_train_len,
                 inner_val_len,
                 outer_val_len,
                 inner_outer_val_len,
                 batch_size,
                 num_workers,
                 context_class_size,
                 minority_group_proportion,
                 are_spurious_tokens_fixed,
                 are_class_tokens_fixed,
                 token_generation_mode,
                 spurious_setting,
                 rotate_encodings,
                 n_rotation_matrices,
                 *args, **kwargs):
        super(INaturalistEmbContextsDataModule, self).__init__()

        # Initializing dataset parameters
        self._dataset_path = dataset_path
        self._encoding_extractor = encoding_extractor
        self._saved_val_sets_path = saved_val_sets_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._context_class_size = context_class_size
        self._minority_group_proportion = minority_group_proportion
        self._are_spurious_tokens_fixed = are_spurious_tokens_fixed
        self._are_class_tokens_fixed = are_class_tokens_fixed
        self._token_generation_mode = token_generation_mode
        self._spurious_setting = spurious_setting
        self._rotate_encodings = rotate_encodings
        self._n_rotation_matrices = n_rotation_matrices

        # Initializing dataset lengths for different splits
        self._inner_train_len = inner_train_len
        self._inner_val_len = inner_val_len
        self._outer_val_len = outer_val_len
        self._inner_outer_val_len = inner_outer_val_len

        # Placeholders for dataset splits
        self._train_set = None
        self._inner_val_set = None
        self._outer_val_set = None
        self._inner_outer_val_set = None

    def setup(self, stage="fit", *args, **kwargs):
        """
        Sets up the dataset splits for training, validation, and testing.
        Initializes the inner train and inner, outer, inner-outer validation datasets.
        """
        # Creating dataset instances for each split using class attributes
        if stage == "fit":
            self._train_set = INaturalistEmbContextsDataset(self._dataset_path, self._encoding_extractor,
                                                            self._inner_train_len,
                                                            "inner_train", "inner_train",
                                                            self._context_class_size,
                                                            self._minority_group_proportion,
                                                            self._are_spurious_tokens_fixed,
                                                            self._are_class_tokens_fixed,
                                                            self._token_generation_mode,
                                                            self._spurious_setting,
                                                            rotate_encodings=self._rotate_encodings,
                                                            n_rotation_matrices=self._n_rotation_matrices)

        saved_data_path = self._saved_val_sets_path and os.path.join(self._saved_val_sets_path,
                                                                     self.ValSets.INNER.value)
        self._inner_val_set = INaturalistEmbContextsDataset(self._dataset_path, self._encoding_extractor,
                                                            self._inner_val_len,
                                                            "inner_val", "inner_val",
                                                            self._context_class_size,
                                                            self._minority_group_proportion,
                                                            self._are_spurious_tokens_fixed,
                                                            self._are_class_tokens_fixed,
                                                            self._token_generation_mode,
                                                            self._spurious_setting,
                                                            saved_data_path=saved_data_path)

        saved_data_path = self._saved_val_sets_path and os.path.join(self._saved_val_sets_path,
                                                                     self.ValSets.OUTER.value)
        self._outer_val_set = INaturalistEmbContextsDataset(self._dataset_path, self._encoding_extractor,
                                                            self._outer_val_len,
                                                            "outer", "outer",
                                                            self._context_class_size, self._minority_group_proportion,
                                                            self._are_spurious_tokens_fixed,
                                                            self._are_class_tokens_fixed,
                                                            self._token_generation_mode,
                                                            self._spurious_setting,
                                                            saved_data_path=saved_data_path)

        saved_data_path = self._saved_val_sets_path and os.path.join(self._saved_val_sets_path,
                                                                     self.ValSets.INNER_OUTER.value)
        self._inner_outer_val_set = INaturalistEmbContextsDataset(self._dataset_path, self._encoding_extractor,
                                                                  self._inner_outer_val_len,
                                                                  "inner_val", "outer",
                                                                  self._context_class_size,
                                                                  self._minority_group_proportion,
                                                                  self._are_spurious_tokens_fixed,
                                                                  self._are_class_tokens_fixed,
                                                                  self._token_generation_mode,
                                                                  self._spurious_setting,
                                                                  saved_data_path=saved_data_path)

    def train_dataloader(self):
        """
        Creates a DataLoader for the inner training set.

        Returns:
            DataLoader: The DataLoader for the inner training set.
        """
        return DataLoader(self._train_set, batch_size=self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self):
        """
        Creates DataLoaders for inner validation and outer sets, and combines them.

        Returns:
            CombinedLoader: A combined DataLoader for "inner", "outer" and "inner-outer" validation sets.
        """
        inner_val_dataloader = DataLoader(self._inner_val_set,
                                          batch_size=self._batch_size,
                                          num_workers=self._num_workers)
        outer_val_dataloader = DataLoader(self._outer_val_set,
                                          batch_size=self._batch_size,
                                          num_workers=self._num_workers)

        inner_outer_val_dataloader = DataLoader(self._inner_outer_val_set,
                                                batch_size=self._batch_size,
                                                num_workers=self._num_workers)

        return CombinedLoader({self.ValSets.INNER: inner_val_dataloader,
                               self.ValSets.OUTER: outer_val_dataloader,
                               self.ValSets.INNER_OUTER: inner_outer_val_dataloader},
                              mode="sequential")
