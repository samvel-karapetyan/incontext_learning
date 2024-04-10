from typing import Optional
import logging

import pytorch_lightning as pl

from enum import Enum
from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader

from src.datamodules.datasets import INaturalistEmbContextsDatasetV2

log = logging.getLogger(__name__)


class INaturalistEmbContextsDataModuleV2(pl.LightningDataModule):
    """A PyTorch Lightning data module for iNaturalist in-context learning instances."""
    class ValSets(Enum):
        """Enum of validation splits of INaturalistEmbContextsDataModuleV2."""
        INNER = "inner"
        OUTER = "outer"
        INNER_OUTER = "inner_outer"

    def __init__(self,
                 dataset_path: str,
                 encoding_extractor: str,
                 saved_val_sets_path: Optional[str],
                 train_len: int,
                 eval_len: int,
                 batch_size: int,
                 num_workers: Optional[int],
                 context_class_size: int,
                 minority_group_proportion: float,
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 v1_behavior: bool,
                 rotate_encodings: bool,
                 n_rotation_matrices: int,
                 label_noise_ratio_interval: list,
                 input_noise_norm_interval: list,
                 permute_input_dim: bool,
                 ask_context_prob: float,
                 *args, **kwargs):
        super(INaturalistEmbContextsDataModuleV2, self).__init__()

        self._core_params = dict(
            dataset_path=dataset_path,
            encoding_extractor=encoding_extractor,
            context_class_size=context_class_size,
            minority_group_proportion=minority_group_proportion,
            spurious_setting=spurious_setting,
            sp_token_generation_mode=sp_token_generation_mode,
            v1_behavior=v1_behavior,
        )

        self._aug_params = dict(
            rotate_encodings=rotate_encodings,
            n_rotation_matrices=n_rotation_matrices,
            label_noise_ratio_interval=label_noise_ratio_interval,
            input_noise_norm_interval=input_noise_norm_interval,
            permute_input_dim=permute_input_dim,
            ask_context_prob=ask_context_prob,
        )

        self._saved_val_sets_path = saved_val_sets_path
        self._train_len = train_len
        self._eval_len = eval_len
        self._batch_size = batch_size
        self._num_workers = num_workers

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
            self._train_set = INaturalistEmbContextsDatasetV2(
                **self._core_params,
                **self._aug_params,
                data_length=self._train_len,
                class1_split="inner_train",
                class2_split="inner_train",
            )

        # saved_data_path = self._saved_val_sets_path and os.path.join(self._saved_val_sets_path,
        #                                                              self.ValSets.INNER.value)
        saved_data_path = None
        self._inner_val_set = INaturalistEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            class1_split="inner_val",
            class2_split="inner_val",
            saved_data_path=saved_data_path)

        # saved_data_path = self._saved_val_sets_path and os.path.join(self._saved_val_sets_path,
        #                                                              self.ValSets.OUTER.value)
        saved_data_path = None
        self._outer_val_set = INaturalistEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            class1_split="outer",
            class2_split="outer",
            saved_data_path=saved_data_path)

        # saved_data_path = self._saved_val_sets_path and os.path.join(self._saved_val_sets_path,
        #                                                              self.ValSets.INNER_OUTER.value)
        saved_data_path = None
        self._inner_outer_val_set = INaturalistEmbContextsDatasetV2(
            **self._core_params,
            data_length=self._eval_len,
            class1_split="inner_val",
            class2_split="outer",
            saved_data_path=saved_data_path)

    def train_dataloader(self):
        return DataLoader(self._train_set, batch_size=self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self):
        """Creates a combined dataloader for all validation datasets.

        Returns:
            CombinedLoader: A combined DataLoader for "inner", "inner-outer", and "outer" validation sets.
        """
        inner_val_dataloader = DataLoader(self._inner_val_set,
                                          batch_size=self._batch_size,
                                          num_workers=self._num_workers)
        inner_outer_val_dataloader = DataLoader(self._inner_outer_val_set,
                                                batch_size=self._batch_size,
                                                num_workers=self._num_workers)
        outer_val_dataloader = DataLoader(self._outer_val_set,
                                          batch_size=self._batch_size,
                                          num_workers=self._num_workers)

        return CombinedLoader({self.ValSets.INNER: inner_val_dataloader,
                               self.ValSets.INNER_OUTER: inner_outer_val_dataloader,
                               self.ValSets.OUTER: outer_val_dataloader},
                              mode="sequential")
