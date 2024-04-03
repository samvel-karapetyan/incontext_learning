from typing import Optional
import logging
import random

import numpy as np

from src.datamodules.datasets.base_emb_contexts_v2 import BaseEmbContextsDatasetV2
from src.datamodules.datasets.waterbirds import CustomizedWaterbirdsDataset as WaterbirdsDataset
from src.utils.dataset_helpers.context_prep_utils import get_group_counts_based_on_proportions

log = logging.getLogger(__name__)

Example = tuple[int, int, int]  # (index, spurious_label, class_label)


class WaterbirdsEmbContextsDatasetV2(BaseEmbContextsDatasetV2):
    """A dataset class for Waterbirds in-context learning instances."""

    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 data_length: int,
                 context_class_size: int,
                 group_proportions: list[float],
                 spurious_setting: str,
                 v1_behavior: bool = False,
                 rotate_encodings: bool = False,
                 n_rotation_matrices: Optional[int] = None,
                 saved_data_path: Optional[str] = None):
        """
        Args:

        root_dir (str): The root directory of the dataset.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_class_size (int): The size of each class in the context.
        group_proportions (list[float]): Proportions for the 4 groups.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'no_spurious'(x), 'sum'(x+c), 'separate_token'(x, c)
                                or 'sum_with_spurious'(x+c, c).
        v1_behavior (bool): Whether intermediate queries should be the context examples.
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
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

        dataset = WaterbirdsDataset(root_dir,
                                    data_type="encoding",
                                    encoding_extractor=encoding_extractor,
                                    return_labels=True)
        groups = np.stack([(2 * y + c) for _, y, c, _ in dataset])

        self._train_set = dataset.get_subset("train")
        self._val_set = dataset.get_subset("val")

        self._train_groups = groups[self._train_set.indices]
        self._val_groups = groups[self._val_set.indices]

    def _generate_context_and_queries(self) -> (list[Example], list[Example]):
        """Samples context and query examples.

        Returns:
            a pair (context, queries), where both are lists of 2 * context_class_size (id, spurious, label) triplets.
        """

        def sample(dataset,
                   dataset_groups,
                   num_examples: int,
                   group_proportions: list[float]) -> list[Example]:
            "Samples a subset of examples given a dataset and groups of its examples."
            group_counts = get_group_counts_based_on_proportions(
                num_examples=num_examples,
                group_proportions=group_proportions)

            indices = []
            for i, group_count in enumerate(group_counts):
                all_group_items = np.where(dataset_groups == i)[0]
                random_items = np.random.choice(all_group_items, group_count, replace=False)
                indices.extend(random_items)

            labels = [dataset[idx][1] for idx in indices]
            spurious_labels = [dataset[idx][2] for idx in indices]
            examples = list(zip(indices, spurious_labels, labels))
            random.shuffle(examples)
            return examples

        context = sample(dataset=self._train_set,
                         dataset_groups=self._train_groups,
                         num_examples=2 * self._context_class_size,
                         group_proportions=self._group_proportions)

        queries = sample(dataset=self._val_set,
                         dataset_groups=self._val_groups,
                         num_examples=2 * self._context_class_size,
                         group_proportions=[0.25, 0.25, 0.25, 0.25])

        return context, queries

    def _prepare_context_image_encodings(
            self,
            context: list[Example]
    ) -> np.ndarray:
        """Returns a matrix of shape [2*C, D] containing context example encodings."""
        return np.stack(
            [self._train_set[idx][0] for idx, _, _ in context]
        )

    def _prepare_query_image_encodings(
            self,
            queries: list[Example]
    ) -> np.ndarray:
        """Returns a matrix of shape [2*C, D] containing query example encodings."""
        return np.stack(
            [self._val_set[idx][0] for idx, _, _ in queries]
        )
