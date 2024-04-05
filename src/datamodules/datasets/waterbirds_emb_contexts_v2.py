from typing import Optional
import logging
import random

import numpy as np

from src.datamodules.datasets.base_emb_contexts_v2 import BaseEmbContextsDatasetV2
from src.datamodules.datasets.waterbirds import WaterbirdsExtracted
from src.utils.dataset_helpers.context_prep_utils import get_group_counts_based_on_proportions

log = logging.getLogger(__name__)

Example = tuple[int, int, int]  # (index, spurious_label, class_label)


def _sample(
        dataset,
        dataset_groups,
        num_examples: int,
        group_proportions: list[float],
        remaining_mask: Optional[np.ndarray] = None,
) -> list[Example]:
    """Samples a subset of examples given a dataset and groups of its examples."""

    if remaining_mask is None:
        remaining_mask = np.ones(len(dataset), dtype=bool)

    group_counts = get_group_counts_based_on_proportions(
        num_examples=num_examples,
        group_proportions=group_proportions)

    indices = []
    for i, group_count in enumerate(group_counts):
        all_group_items = np.where(remaining_mask & (dataset_groups == i))[0]
        random_items = np.random.choice(all_group_items, group_count, replace=False)
        indices.extend(random_items)

    labels = [dataset[idx][1] for idx in indices]
    spurious_labels = [dataset[idx][2] for idx in indices]
    examples = list(zip(indices, spurious_labels, labels))
    random.shuffle(examples)
    return examples


class WaterbirdsEmbContextsDatasetV2(BaseEmbContextsDatasetV2):
    """A dataset class for Waterbirds in-context learning instances."""

    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 data_length: int,
                 context_split: str,
                 query_split: str,
                 context_class_size: int,
                 group_proportions: list[float],
                 spurious_setting: str,
                 v1_behavior: bool = False,
                 rotate_encodings: bool = False,
                 n_rotation_matrices: Optional[int] = None,
                 randomly_swap_labels: bool = False,
                 add_context_noise: bool = False,
                 saved_data_path: Optional[str] = None):
        """
        Args:

        root_dir (str): The root directory of the dataset.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_split (str): The split where context examples are selected from ('train' or 'val').
        query_split (str): The split where query examples are selected from ('train' or 'val').
        context_class_size (int): The size of each class in the context.
        group_proportions (list[float]): Proportions for the 4 groups.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'no_spurious'(x), 'sum'(x+c), 'separate_token'(x, c)
                                or 'sum_with_spurious'(x+c, c).
        v1_behavior (bool): Whether intermediate queries should be the context examples.
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
        randomly_swap_labels (bool): Whether to randomly swap labels (0 -> 1 and 1 -> 0) when creating an ILC instance.
        add_context_noise (bool): Whether to randomly add input or label noise to the context.
        saved_data_path (str or None): Path for loading data; if None, new data is generated.
        """
        super(WaterbirdsEmbContextsDatasetV2, self).__init__(
            encoding_extractor=encoding_extractor,
            data_length=data_length,
            context_class_size=context_class_size,
            spurious_setting=spurious_setting,
            v1_behavior=v1_behavior,
            rotate_encodings=rotate_encodings,
            n_rotation_matrices=n_rotation_matrices,
            saved_data_path=saved_data_path,
        )

        self._group_proportions = group_proportions
        self._randomly_swap_labels = randomly_swap_labels
        self._add_context_noise = add_context_noise

        dataset = WaterbirdsExtracted(root_dir,
                                      encoding_extractor=encoding_extractor)

        train_set = dataset.get_subset("train")
        train_groups = np.stack([(2 * y + c) for _, y, c, _ in train_set])

        val_set = dataset.get_subset("val")
        val_groups = np.stack([(2 * y + c) for _, y, c, _ in val_set])

        self._context_split = context_split
        if context_split == 'train':
            self._context_set = train_set
            self._context_groups = train_groups
        elif context_split == 'val':
            self._context_set = val_set
            self._context_groups = val_groups
        else:
            raise ValueError()

        self._query_split = query_split
        if query_split == 'train':
            self._query_set = train_set
            self._query_groups = train_groups
        elif query_split == 'val':
            self._query_set = val_set
            self._query_groups = val_groups
        else:
            raise ValueError()

    def _generate_context_and_queries(self) -> (list[Example], list[Example]):
        """Samples context and query examples.

        Returns:
            a pair (context, queries), where both are lists of 2 * context_class_size (id, spurious, label) triplets.
        """

        context = _sample(dataset=self._context_set,
                          dataset_groups=self._context_groups,
                          num_examples=2 * self._context_class_size,
                          group_proportions=self._group_proportions)

        if self._context_split == self._query_split:
            remaining_mask = np.ones(len(self._context_set), dtype=bool)
            for idx, _, _ in context:
                remaining_mask[idx] = False
        else:
            remaining_mask = None

        queries = _sample(dataset=self._query_set,
                          dataset_groups=self._query_groups,
                          num_examples=2 * self._context_class_size,
                          group_proportions=[0.25, 0.25, 0.25, 0.25],
                          remaining_mask=remaining_mask)

        if self._randomly_swap_labels and np.random.rand() < 0.5:
            context = [(idx, sp, 1 - label) for idx, sp, label in context]
            queries = [(idx, sp, 1 - label) for idx, sp, label in queries]

        if self._add_context_noise:
            label_noise_count = int(np.random.uniform(low=0, high=0.2) * len(context)) 
            label_noise_indices = np.random.choice(len(context), label_noise_count, replace=False)

            for i in label_noise_indices:
                idx, sp, label = context[i]
                context[i] = (idx, sp, 1 - label)

        return context, queries

    def _prepare_context_image_encodings(
            self,
            context: list[Example]
    ) -> np.ndarray:
        """Returns a matrix of shape [2*C, D] containing context example encodings."""
        image_encodings =  np.stack(
            [self._context_set[idx][0] for idx, _, _ in context]
        )

        if self._add_context_noise:
            input_noise_ratio = np.random.uniform(low=0, high=0.5)
            input_noise = np.random.normal(size=image_encodings.shape).astype(image_encodings.dtype)

            input_noise = input_noise / np.linalg.norm(input_noise, axis=1, keepdims=True) \
                * np.linalg.norm(image_encodings, axis=1, keepdims=True) * input_noise_ratio
            
            image_encodings = image_encodings + input_noise

        return image_encodings

    def _prepare_query_image_encodings(
            self,
            queries: list[Example]
    ) -> np.ndarray:
        """Returns a matrix of shape [2*C, D] containing query example encodings."""
        return np.stack(
            [self._query_set[idx][0] for idx, _, _ in queries]
        )
