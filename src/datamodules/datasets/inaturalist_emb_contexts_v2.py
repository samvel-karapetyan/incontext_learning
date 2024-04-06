from typing import Optional
import logging
import os
import pandas as pd
import numpy as np
import random

from src.datamodules.datasets.base_emb_contexts_v2 import BaseEmbContextsDatasetV2
from src.utils.dataset_helpers.context_prep_utils import generate_spurious_labels, prepare_context_or_query

log = logging.getLogger(__name__)

Example = tuple[int, int, int]  # (index, spurious_label, class_label)


class INaturalistEmbContextsDatasetV2(BaseEmbContextsDatasetV2):
    """A dataset class for iNaturalist in-context learning instances."""

    def __init__(self,
                 dataset_path: str,
                 encoding_extractor: str,
                 data_length: int,
                 class1_split: str,
                 class2_split: str,
                 context_class_size: int,
                 minority_group_proportion: float,
                 spurious_setting: str,
                 v1_behavior: bool = False,
                 rotate_encodings: bool = False,
                 n_rotation_matrices: Optional[int] = None,
                 label_noise_ratio_interval: Optional[list] = None,
                 input_noise_std_interval: Optional[list] = None,
                 permute_input_dim: bool = False,
                 ask_context_prob: Optional[float] = None,
                 saved_data_path: Optional[str] = None,
                 ):
        """
        Arguments:
        dataset_path (str): The path to the dataset directory.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        class1_split (str): The type of data split of class 1 (e.g., 'inner_train', 'inner_val', 'outer').
        class2_split (str): The type of data split of class 2 (e.g., 'inner_train', 'inner_val', 'outer').
        context_class_size (int): The size of each class in the context.
        minority_group_proportion (float): The proportion of the minority group in the context per class.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'no_spurious'(x), 'sum'(x+c), 'separate_token'(x, c)
                                or 'sum_with_spurious'(x+c, c).
        v1_behavior (bool): Whether intermediate queries should be the context examples.
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
        label_noise_ratio_interval (list or None): Interval for the ratio of label noise. 
                                If None, no label noise is added.
        input_noise_std_interval (list or None): Interval for the standard deviation of Gaussian noise.
                                If None, no Gaussian noise is added to representations.
        permute_input_dim (bool): Determines if image encodings are permuted. 
                                True enables permutation, while False bypasses it.
        ask_context_prob (float or None). If specified, defines the probability with which a query is set to be one
                                          of previous context examples.
        saved_data_path (str or None): Path for loading data; if None, new data is generated.
        """
        super(INaturalistEmbContextsDatasetV2, self).__init__(
            encoding_extractor=encoding_extractor,
            data_length=data_length,
            context_class_size=context_class_size,
            spurious_setting=spurious_setting,
            v1_behavior=v1_behavior,
            rotate_encodings=rotate_encodings,
            n_rotation_matrices=n_rotation_matrices,
            label_noise_ratio_interval=label_noise_ratio_interval,
            input_noise_std_interval=input_noise_std_interval,
            permute_input_dim=permute_input_dim,
            ask_context_prob=ask_context_prob,
            saved_data_path=saved_data_path,
        )

        # Prepare encodings and data files
        encodings_data = np.load(os.path.join(dataset_path, encoding_extractor, "combined.npz"))
        self._encodings = encodings_data["encodings"]
        self._encodings_indices_map = encodings_data["indices_map"]

        self._dataframe = pd.read_csv(os.path.join(dataset_path, "prepared_data.csv"))
        self._dataframe = self._dataframe[self._dataframe['split'].isin([class1_split, class2_split])]

        # Unique categories in the dataset
        self._class1_split = class1_split
        self._class2_split = class2_split
        if class1_split == class2_split:
            self._categories = self._dataframe["category"].unique()
        else:
            self._categories = (self._dataframe.loc[self._dataframe['split'] == class1_split, "category"].unique(),
                                self._dataframe.loc[self._dataframe['split'] == class2_split, "category"].unique())

        self._minority_group_proportion = minority_group_proportion

    def _generate_context_and_queries(self) -> (list[Example], list[Example]):
        """Samples context and query examples.

        Returns:
            a pair (context, queries), where both are lists of 2 * context_class_size (id, spurious, label) triplets.
        """
        # Randomly selecting two categories
        if self._class1_split != self._class2_split:
            category1, category2 = (np.random.choice(x, size=1)[0] for x in self._categories)
        else:
            category1, category2 = np.random.choice(self._categories, size=2, replace=False)

        # Sampling 2 * context_class_size examples from each category (half for context and half for queries)
        df = self._dataframe  # shorthand
        cat1_indices = df.loc[df["category"] == category1, "image_id"].sample(2 * self._context_class_size).tolist()
        cat2_indices = df.loc[df["category"] == category2, "image_id"].sample(2 * self._context_class_size).tolist()
        cat1_context_indices = cat1_indices[:self._context_class_size]
        cat1_query_indices = cat1_indices[self._context_class_size:]
        cat2_context_indices = cat2_indices[:self._context_class_size]
        cat2_query_indices = cat2_indices[self._context_class_size:]

        # Shuffling class and spurious labels
        class_labels, spurious_labels = [0, 1], [0, 1]
        random.shuffle(class_labels)
        random.shuffle(spurious_labels)
        cat1_class_label, cat2_class_label = class_labels
        spurious_label1, spurious_label2 = spurious_labels

        # Generate spurious labels for context examples
        cat1_context_spurious_labels = generate_spurious_labels(
            spurious_label1, spurious_label2, self._context_class_size, self._minority_group_proportion)
        cat2_context_spurious_labels = generate_spurious_labels(
            spurious_label2, spurious_label1, self._context_class_size, self._minority_group_proportion)

        # Generate spurious labels for queries. Here we use balanced sampling (no minority / majority).
        cat1_query_spurious_labels = generate_spurious_labels(
            spurious_label1, spurious_label2, self._context_class_size, 0.5)
        cat2_query_spurious_labels = generate_spurious_labels(
            spurious_label2, spurious_label1, self._context_class_size, 0.5)

        # Prepare full context information
        context = prepare_context_or_query(
            cat1_indices=cat1_context_indices,
            cat2_indices=cat2_context_indices,
            cat1_spurious_labels=cat1_context_spurious_labels,
            cat2_spurious_labels=cat2_context_spurious_labels,
            cat1_class_label=cat1_class_label,
            cat2_class_label=cat2_class_label)

        queries = prepare_context_or_query(
            cat1_indices=cat1_query_indices,
            cat2_indices=cat2_query_indices,
            cat1_spurious_labels=cat1_query_spurious_labels,
            cat2_spurious_labels=cat2_query_spurious_labels,
            cat1_class_label=cat1_class_label,
            cat2_class_label=cat2_class_label)

        return context, queries

    def _prepare_image_encodings(
            self,
            examples: list[Example],
    ) -> np.ndarray:
        """
        Transforms image encodings based on their labels using the appropriate encoding transformer.

        Args:
            examples: List of tuples for the context images containing IDs, spurious labels, and class labels.

        Returns: np.ndarray of shape (num_examples, dim).
        """
        img_encodings = []
        for example in examples:
            image_id = example[0]
            img_encodings.append(self._encodings[self._encodings_indices_map[image_id]])
        return np.stack(img_encodings)

    def _prepare_context_image_encodings(
            self,
            context: list[Example]
    ) -> np.ndarray:
        return self._prepare_image_encodings(context)

    def _prepare_query_image_encodings(
            self,
            queries: list[Example]
    ) -> np.ndarray:
        return self._prepare_image_encodings(queries)
