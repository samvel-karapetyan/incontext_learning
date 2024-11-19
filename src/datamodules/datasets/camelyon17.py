import dataclasses
from typing import Optional
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
class CustomExtractedWILDSSubset:
    encodings: np.ndarray
    y_array: torch.Tensor
    metadata_array: torch.Tensor

    def __len__(self):
        return len(self.y_array)


class Camelyon17SubsetExtracted(Dataset):
    def __init__(self,
                 custom_subset: CustomExtractedWILDSSubset,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self._custom_subset = custom_subset
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add

    def __getitem__(self, indices) -> (np.ndarray, Examples):
        x = self._custom_subset.encodings[indices].copy()
        y = self._custom_subset.y_array[indices].numpy()

        # metadata fields: ['hospital', 'slide', 'y', 'from_source_domain']
        hospital_id = self._custom_subset.metadata_array[indices, 0].numpy()
        # In our constructed train and val splits, we have hospitals 4 (from original train) and 2 (from original test)
        # In our constructed test split, we have hospitals 0 (from original train 1) and 1 (from original val)
        # Therefore, we map hospitals 1 and 4 to spurious=0 and hospitals 1 and 2 to spurious=1.
        c = ((hospital_id == 1) | (hospital_id == 2)).astype(np.int32)

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
        return len(self._custom_subset)


def _put_encodings_in_right_order(
        indices_within_full_camelyon17: np.ndarray,
        encodings: np.ndarray,
        index_map: np.ndarray,
) -> np.ndarray:
    """Permutes rows of `encodings` such that the i-th row corresponds to the i-th example of the subset."""
    n = len(indices_within_full_camelyon17)
    row_indices = np.zeros(n, dtype=np.int32)
    for idx in range(n):
        idx_within_full_camelyon17 = indices_within_full_camelyon17[idx],
        encoding_row_index = index_map[idx_within_full_camelyon17]
        assert encoding_row_index != -1
        row_indices[idx] = encoding_row_index
    return encodings[row_indices]


def _select_hospitals(
        ds: CustomExtractedWILDSSubset,
        selected_hospital_ids: list[int],
) -> CustomExtractedWILDSSubset:
    hospital_ids = ds.metadata_array[:, 0].numpy()
    mask = np.zeros(hospital_ids.shape, dtype=np.bool_)
    for h in selected_hospital_ids:
        mask |= (hospital_ids == h)
    return CustomExtractedWILDSSubset(
        y_array=ds.y_array[mask],
        metadata_array=ds.metadata_array[mask],
        encodings=ds.encodings[mask],
    )


def _concatenate_custom_wilds_subsets(
        part1: CustomExtractedWILDSSubset,
        part2: CustomExtractedWILDSSubset,
) -> CustomExtractedWILDSSubset:
    return CustomExtractedWILDSSubset(
        y_array=torch.concat(
            [part1.y_array, part2.y_array], dim=0
        ),
        metadata_array=torch.concat(
            [part1.metadata_array, part2.metadata_array], dim=0
        ),
        encodings=np.concatenate(
            [part1.encodings, part2.encodings], axis=0
        )
    )


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

    def _split_wilds_train_val(self,
                               wilds_split: str,
                               train_ratio: float,
                               ) -> (CustomExtractedWILDSSubset, CustomExtractedWILDSSubset):
        encodings_data = np.load(
            os.path.join(self._root_dir, "camelyon17", self._encoding_extractor, wilds_split, "combined.npz"))
        wilds_subset = self._wilds_camelyon17.get_subset(wilds_split)

        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(len(wilds_subset))
        train_count = int(train_ratio * len(wilds_subset))
        train_indices = perm[:train_count]
        val_indices = perm[train_count:]

        encodings = _put_encodings_in_right_order(
            indices_within_full_camelyon17=wilds_subset.indices,
            encodings=encodings_data['encodings'],
            index_map=encodings_data['indices_map'],
        )

        train_ds = CustomExtractedWILDSSubset(
            y_array=wilds_subset.y_array[train_indices],
            metadata_array=wilds_subset.metadata_array[train_indices],
            encodings=encodings[train_indices],
        )

        val_ds = CustomExtractedWILDSSubset(
            y_array=wilds_subset.y_array[val_indices],
            metadata_array=wilds_subset.metadata_array[val_indices],
            encodings=encodings[val_indices],
        )

        return train_ds, val_ds

    def _select_wilds_subset(self,
                             wilds_split: str) -> CustomExtractedWILDSSubset:
        encodings_data = np.load(
            os.path.join(self._root_dir, "camelyon17", self._encoding_extractor, wilds_split, "combined.npz"))
        wilds_subset = self._wilds_camelyon17.get_subset(wilds_split)

        encodings = _put_encodings_in_right_order(
            indices_within_full_camelyon17=wilds_subset.indices,
            encodings=encodings_data['encodings'],
            index_map=encodings_data['indices_map'],
        )

        return CustomExtractedWILDSSubset(
            y_array=wilds_subset.y_array,
            metadata_array=wilds_subset.metadata_array,
            encodings=encodings,
        )

    def get_subset(self, split, *args, **kwargs) -> Camelyon17SubsetExtracted:
        # 3 splits are allowed: train, val, and test
        # * train and val are constructed by taking hospital 4 of the original train and hospital 2 of the
        #   original test sets, then splitting by 80% vs 20% ratio.
        # * test is constructed by taking the hospital  0 of the original train and hospital 1 of the original
        #   OOD validation.
        if split == 'train':
            part1, _ = self._split_wilds_train_val(
                wilds_split='train',
                train_ratio=0.8,
            )
            part1 = _select_hospitals(
                part1, selected_hospital_ids=[4])  # 105,821 examples of hospital 4

            part2, _ = self._split_wilds_train_val(
                wilds_split='test',
                train_ratio=0.8,
            )  # 68,043 examples of hospital 2

            joint = _concatenate_custom_wilds_subsets(part1, part2)

            return Camelyon17SubsetExtracted(
                custom_subset=joint,
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add,
            )

        if split == 'val':
            _, part1 = self._split_wilds_train_val(
                wilds_split='train',
                train_ratio=0.8,
            )
            part1 = _select_hospitals(
                part1, selected_hospital_ids=[4])  # 26,231 examples of hospital 4

            _, part2 = self._split_wilds_train_val(
                wilds_split='test',
                train_ratio=0.8,
            )  # 17,011 examples of hospital 2

            joint = _concatenate_custom_wilds_subsets(part1, part2)

            return Camelyon17SubsetExtracted(
                custom_subset=joint,
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add,
            )

        if split == 'test':
            part1 = self._select_wilds_subset(
                wilds_split='train',
            )
            part1 = _select_hospitals(
                part1, selected_hospital_ids=[0])  # 53,425 examples of hospital 0

            part2 = self._select_wilds_subset(
                wilds_split='val',
            )  # 34,904 examples of hospital 1

            joint = _concatenate_custom_wilds_subsets(part1, part2)

            return Camelyon17SubsetExtracted(
                custom_subset=joint,
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add,
            )

        raise ValueError(f'Unexpected value {split=}')
