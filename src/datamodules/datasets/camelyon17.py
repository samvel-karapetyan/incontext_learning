import dataclasses
from typing import Optional, Union
import logging
import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.wilds_dataset import WILDSSubset

log = logging.getLogger(__name__)

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


class Camelyon17SubsetForEncodingExtraction(Dataset):
    def __init__(self, wilds_camelyon17_subset: WILDSSubset):
        self._wilds_camelyon17_subset = wilds_camelyon17_subset

        if hasattr(self._wilds_camelyon17_subset, 'collate'):  # necessary method for constructing subsets
            setattr(self, 'collate', getattr(self._wilds_camelyon17_subset, 'collate'))

    def __getitem__(self, idx):
        x, y, metadata = self._wilds_camelyon17_subset[idx]
        return x, self._wilds_camelyon17_subset.indices[idx]

    def __len__(self):
        return len(self._wilds_camelyon17_subset)


class Camelyon17ForEncodingExtraction:
    def __init__(self, root_dir: str, download: bool = False):
        self._wilds_camelyon17 = Camelyon17Dataset(root_dir=root_dir, download=download)

    def get_subset(self, *args, **kwargs) -> Camelyon17SubsetForEncodingExtraction:
        return Camelyon17SubsetForEncodingExtraction(
            self._wilds_camelyon17.get_subset(*args, **kwargs))


@dataclasses.dataclass
class MockWILDSSubset:
    indices: np.ndarray
    y_array: torch.Tensor
    metadata_array: torch.Tensor

    def __len__(self):
        return len(self.y_array)


class Camelyon17SubsetExtracted(Dataset):
    def __init__(self,
                 wilds_camelyon17_subset: Union[WILDSSubset, MockWILDSSubset],
                 encodings: np.ndarray,
                 index_map: np.ndarray,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self._wilds_camelyon17_subset = wilds_camelyon17_subset
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add

        # permute rows of `encodings` such that the i-th row corresponds to the i-th example of the subset
        n = len(wilds_camelyon17_subset)
        row_indices = np.zeros(n, dtype=np.int32)
        for idx in range(n):
            idx_within_full_camelyon17 = wilds_camelyon17_subset.indices[idx]
            encoding_row_index = index_map[idx_within_full_camelyon17]
            assert encoding_row_index != -1
            row_indices[idx] = encoding_row_index
        self._encodings = encodings[row_indices]

    def __getitem__(self, indices) -> (np.ndarray, Examples):
        x = self._encodings[indices].copy()
        y = self._wilds_camelyon17_subset.y_array[indices].numpy()

        # metadata fields: ['hospital', 'slide', 'y', 'from_source_domain']
        hospital_id = self._wilds_camelyon17_subset.metadata_array[indices, 0].numpy()
        # Training and ID validation hospitals: 0, 3 and 4
        # OOD validation hospitals: 1
        # Test hospitals: 2
        # NOTE: We set the spurious variable based on the hospital ID.
        c = ((hospital_id == 4) | (hospital_id == 2)).astype(np.int32)

        # add more background information if specified
        if self._sp_vector_to_add is not None:
            x += np.outer(2 * c - 1, self._sp_vector_to_add)

        # reverse the task if specified
        if not self._reverse_task:
            examples = np.stack([indices, c, y], axis=1)
        else:
            examples = np.stack([indices, y, c], axis=1)

        return x, examples

    def __len__(self):
        return len(self._wilds_camelyon17_subset)


class Camelyon17Extracted:
    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self._root_dir = root_dir
        self._encoding_extractor = encoding_extractor
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add
        self._wilds_camelyon17 = Camelyon17Dataset(root_dir=root_dir)

    def get_subset(self, split, *args, **kwargs) -> Camelyon17SubsetExtracted:
        if split in ['train', 'val']:
            if split == 'train':
                wilds_split = 'train'
            else:
                wilds_split = 'id_val'
            encodings_data = np.load(
                os.path.join(self._root_dir, "camelyon17", self._encoding_extractor, wilds_split, "combined.npz"))
            return Camelyon17SubsetExtracted(
                wilds_camelyon17_subset=self._wilds_camelyon17.get_subset(wilds_split, *args, **kwargs),
                encodings=encodings_data['encodings'],
                index_map=encodings_data['indices_map'],
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add,
            )
        if split == 'test':
            val_encodings_data = np.load(
                os.path.join(self._root_dir, "camelyon17", self._encoding_extractor, "val", "combined.npz"))

            val_subset = Camelyon17SubsetExtracted(
                wilds_camelyon17_subset=self._wilds_camelyon17.get_subset('val', *args, **kwargs),
                encodings=val_encodings_data['encodings'],
                index_map=val_encodings_data['indices_map'],
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add,
            )

            test_encodings_data = np.load(
                os.path.join(self._root_dir, "camelyon17", self._encoding_extractor, "test", "combined.npz"))

            test_subset = Camelyon17SubsetExtracted(
                wilds_camelyon17_subset=self._wilds_camelyon17.get_subset('test', *args, **kwargs),
                encodings=test_encodings_data['encodings'],
                index_map=test_encodings_data['indices_map'],
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add,
            )

            joint_encodings = np.concatenate(
                [val_encodings_data['encodings'], test_encodings_data['encodings']],
                axis=0)

            val_index_map = val_encodings_data['indices_map']
            test_index_map = test_encodings_data['indices_map']
            joint_index_map = np.full(shape=max(len(val_index_map), len(test_index_map)),
                                      fill_value=-1)
            joint_index_map[:len(val_index_map)] = val_index_map
            for i in range(len(test_index_map)):
                if test_index_map[i] != -1:
                    assert joint_index_map[i] == -1
                    joint_index_map[i] = len(val_subset) + test_index_map[i]

            joint_y_array = torch.concat(
                [val_subset._wilds_camelyon17_subset.y_array,
                 test_subset._wilds_camelyon17_subset.y_array],
                dim=0)

            joint_metadata_array = torch.concat(
                [val_subset._wilds_camelyon17_subset.metadata_array,
                 test_subset._wilds_camelyon17_subset.metadata_array],
                dim=0)

            joint_indices_array = np.concatenate(
                [val_subset._wilds_camelyon17_subset.indices,
                 test_subset._wilds_camelyon17_subset.indices],
                axis=0)

            return Camelyon17SubsetExtracted(
                wilds_camelyon17_subset=MockWILDSSubset(
                    y_array=joint_y_array,
                    metadata_array=joint_metadata_array,
                    indices=joint_indices_array,
                ),
                encodings=joint_encodings,
                index_map=joint_index_map,
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add)

        raise ValueError(f'Unexpected value {split=}')
