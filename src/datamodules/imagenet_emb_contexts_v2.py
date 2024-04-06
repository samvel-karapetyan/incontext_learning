from typing import Optional
import logging
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.datamodules.datasets import ImagenetEmbContextsDatasetV2

log = logging.getLogger(__name__)


class ImagenetEmbContextsDataModuleV2(pl.LightningDataModule):
    """A PyTorch Lightning data module for ImageNet in-context learning instances.

    TODO(hrayr): extend to proper training.
    """

    def __init__(self,
                 dataset_path: str,
                 encoding_extractor: str,
                 data_length: int,
                 saved_data_path: Optional[str],
                 batch_size: int,
                 num_workers: Optional[int],
                 context_class_size: int,
                 minority_group_proportion: float,
                 spurious_setting: str,
                 v1_behavior: bool,
                 rotate_encodings: bool,
                 n_rotation_matrices: int,
                 label_noise_ratio_interval: list,
                 input_noise_std_interval: list,
                 permute_input_dim: bool,
                 *args, **kwargs):
        super(ImagenetEmbContextsDataModuleV2, self).__init__()

        # Initializing dataset parameters
        self._dataset_path = dataset_path
        self._encoding_extractor = encoding_extractor
        self._data_length = data_length
        self._saved_data_path = saved_data_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._context_class_size = context_class_size
        self._minority_group_proportion = minority_group_proportion
        self._spurious_setting = spurious_setting
        self._v1_behavior = v1_behavior
        self._rotate_encodings = rotate_encodings
        self._n_rotation_matrices = n_rotation_matrices
        self._label_noise_ratio_interval = label_noise_ratio_interval
        self._input_noise_std_interval = input_noise_std_interval
        self._permute_input_dim = permute_input_dim

        self._dataset = None

    def setup(self,  *args, **kwargs):
        """
        Initializes the dataset.
        """
        # Creating dataset instances for each split using class attributes
        self._dataset = ImagenetEmbContextsDatasetV2(
            dataset_path=self._dataset_path,
            encoding_extractor=self._encoding_extractor,
            data_length=self._data_length,
            context_class_size=self._context_class_size,
            minority_group_proportion=self._minority_group_proportion,
            spurious_setting=self._spurious_setting,
            v1_behavior=self._v1_behavior,
            rotate_encodings=self._rotate_encodings,
            n_rotation_matrices=self._n_rotation_matrices,
            label_noise_ratio_interval=self._label_noise_ratio_interval,
            input_noise_std_interval=self._input_noise_std_interval,
            permute_input_dim=self._permute_input_dim,
            saved_data_path=self._saved_data_path)

    def val_dataloader(self):
        return DataLoader(self._dataset, batch_size=self._batch_size, num_workers=self._num_workers)
