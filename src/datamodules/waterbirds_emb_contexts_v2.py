from typing import Optional
import logging

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from src.datamodules.datasets import WaterbirdsEmbContextsDatasetV2

log = logging.getLogger(__name__)


class WaterbirdsEmbContextsDataModuleV2(pl.LightningDataModule):
    """A PyTorch Lightning data module for Waterbirds in-context learning instances.

    TODO(hrayr): extend to proper training.
    """

    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 data_length: int,
                 saved_data_path: Optional[str],
                 batch_size: int,
                 num_workers: Optional[int],
                 context_class_size: int,
                 group_proportions: list[float],
                 spurious_setting: str,
                 v1_behavior: bool,
                 rotate_encodings: bool,
                 n_rotation_matrices: int,
                 *args, **kwargs):
        super(WaterbirdsEmbContextsDataModuleV2, self).__init__()

        # Initializing dataset parameters
        self._root_dir = root_dir
        self._encoding_extractor = encoding_extractor
        self._data_length = data_length
        self._saved_data_path = saved_data_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._context_class_size = context_class_size
        self._group_proportions = group_proportions
        self._spurious_setting = spurious_setting
        self._v1_behavior = v1_behavior
        self._rotate_encodings = rotate_encodings
        self._n_rotation_matrices = n_rotation_matrices

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
            rotate_encodings=self._rotate_encodings,
            n_rotation_matrices=self._n_rotation_matrices,
            saved_data_path=self._saved_data_path)

    def val_dataloader(self):
        return DataLoader(self._dataset, batch_size=self._batch_size, num_workers=self._num_workers)
