from typing import Optional
import logging
import os

import pandas as pd
import numpy as np

from src.datamodules.datasets import ImagenetDataset
from src.datamodules.datasets.base_emb_contexts_v2 import BaseEmbContextsDatasetV2
from src.utils.dataset_helpers.context_prep_utils import generate_spurious_labels, prepare_context_or_query

log = logging.getLogger(__name__)

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


class ImagenetEmbContextsDatasetV2(BaseEmbContextsDatasetV2):
    """A dataset class for ImageNet in-context learning instances."""

    def __init__(self,
                 dataset_path: str,
                 encoding_extractor: str,
                 data_length: int,
                 context_class_size: int,
                 context_minority_group_proportion: float,
                 query_minority_group_proportion: float,
                 spurious_setting: str,
                 sp_token_generation_mode: str,
                 use_context_as_intermediate_queries: bool = False,
                 rotate_encodings: bool = False,
                 n_rotation_matrices: Optional[int] = None,
                 label_noise_ratio_interval: Optional[list] = None,
                 input_noise_norm_interval: Optional[list] = None,
                 permute_input_dim: bool = False,
                 ask_context_prob: Optional[float] = None,
                 ):
        """
        Arguments:
        dataset_path (str): The path to the dataset directory.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_class_size (int): The size of each class in the context.
        context_minority_group_proportion (float): The proportion of the minority group in the context per class.
        query_minority_group_proportion (float): The proportion of the minority group in the context per class.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
        sp_token_generation_mode (str): Specifies whether the representations of two spurious labels should be
                                        'opposite' or 'random'.
        use_context_as_intermediate_queries (bool): Whether intermediate queries should be the context examples.
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
        label_noise_ratio_interval (list or None): Interval for the ratio of label noise. 
                                If None, no label noise is added.
        input_noise_norm_interval (list or None): Interval for the norm of Gaussian noise.
                                If None, no Gaussian noise is added to representations.
        permute_input_dim (bool): Determines if image encodings are permuted. 
                                True enables permutation, while False bypasses it.
        ask_context_prob (float or None). If specified, defines the probability with which a query is set to be one
                                          of previous context examples.
        """
        super(ImagenetEmbContextsDatasetV2, self).__init__(
            encoding_extractor=encoding_extractor,
            data_length=data_length,
            context_class_size=context_class_size,
            spurious_setting=spurious_setting,
            sp_token_generation_mode=sp_token_generation_mode,
            use_context_as_intermediate_queries=use_context_as_intermediate_queries,
            rotate_encodings=rotate_encodings,
            n_rotation_matrices=n_rotation_matrices,
            label_noise_ratio_interval=label_noise_ratio_interval,
            input_noise_norm_interval=input_noise_norm_interval,
            permute_input_dim=permute_input_dim,
            ask_context_prob=ask_context_prob,
        )

        # Prepare encodings and data files
        encodings_data = np.load(
            os.path.join(dataset_path, encoding_extractor, 'train', "combined.npz"))
        self._encodings = encodings_data["encodings"]
        self._encodings_indices_map = encodings_data["indices_map"]

        self._dataframe = pd.read_csv(os.path.join(dataset_path, ImagenetDataset.annotation_files['train']), sep=' ',
                                      header=None, names=['filepath', 'id'])

        self._dataframe["category"] = self._dataframe["filepath"].apply(lambda x: x.split('/')[0])

        # Calculate the value counts of categories that appear more than context_class_size times
        valid_categories = self._dataframe['category'].value_counts()[lambda x: x > context_class_size].index

        # Filter the dataframe to only include rows with those categories
        self._dataframe = self._dataframe[self._dataframe['category'].isin(valid_categories)]

        self._categories = self._dataframe["category"].unique()

        self._context_minority_group_proportion = context_minority_group_proportion
        self._query_minority_group_proportion = query_minority_group_proportion

    def _generate_context_and_queries(
            self,
            num_context_examples: int,
            num_query_examples: int,
    ) -> (Examples, Examples):
        """Samples context and query examples.

        Returns:
            a pair (context, queries), where both are of type Examples.
        """
        # Randomly selecting two categories
        category1, category2 = np.random.choice(self._categories, size=2, replace=False)

        context_cat1_size = num_context_examples // 2
        context_cat2_size = (num_context_examples + 1) // 2
        query_cat1_size = num_query_examples // 2
        query_cat2_size = (num_query_examples + 1) // 2

        # Sampling 2 * context_class_size examples from each category (half for context and half for queries)
        df = self._dataframe  # shorthand
        cat1_indices = df.loc[df["category"] == category1, "id"].sample(
            context_cat1_size + query_cat1_size).tolist()
        cat2_indices = df.loc[df["category"] == category2, "id"].sample(
            context_cat2_size + query_cat2_size).tolist()
        cat1_context_indices = cat1_indices[:context_cat1_size]
        cat1_query_indices = cat1_indices[context_cat1_size:]
        cat2_context_indices = cat2_indices[:context_cat2_size]
        cat2_query_indices = cat2_indices[context_cat2_size:]

        # Shuffling class and spurious labels
        class_labels, spurious_labels = [0, 1], [0, 1]
        cat1_class_label, cat2_class_label = class_labels
        spurious_label1, spurious_label2 = spurious_labels

        # Generate spurious labels for context examples
        cat1_context_spurious_labels = generate_spurious_labels(
            spurious_label1, spurious_label2, context_cat1_size, self._context_minority_group_proportion)
        cat2_context_spurious_labels = generate_spurious_labels(
            spurious_label2, spurious_label1, context_cat2_size, self._context_minority_group_proportion)

        # Generate spurious labels for queries
        cat1_query_spurious_labels = generate_spurious_labels(
            spurious_label1, spurious_label2, query_cat1_size, self._query_minority_group_proportion)
        cat2_query_spurious_labels = generate_spurious_labels(
            spurious_label2, spurious_label1, query_cat2_size, self._query_minority_group_proportion)

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
            examples: Examples,
    ) -> np.ndarray:
        """
        Transforms image encodings based on their labels using the appropriate encoding transformer.

        Args:
            examples: List of tuples for the context images containing IDs, spurious labels, and class labels.

        Returns: np.ndarray of shape (num_examples, dim).
        """
        indices = examples[:, 0]
        return self._encodings[self._encodings_indices_map[indices]]

    def _prepare_context_image_encodings(
            self,
            context: Examples,
    ) -> np.ndarray:
        return self._prepare_image_encodings(context)

    def _prepare_query_image_encodings(
            self,
            queries: Examples,
    ) -> np.ndarray:
        return self._prepare_image_encodings(queries)
