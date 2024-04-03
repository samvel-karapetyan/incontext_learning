from typing import Optional
import logging
import os
import pandas as pd
import numpy as np
import random

from torch.utils.data import Dataset
from src.utils.dataset_helpers import TokenGenerator
from src.utils.dataset_helpers.encoding_transforms import EncodingRotator, IdentityTransform
from src.utils.dataset_helpers.context_prep_utils import generate_spurious_labels, get_context_example_tokens,\
    get_query_example_tokens

log = logging.getLogger(__name__)

Example = tuple[int, int, int]  # (index, spurious_label, class_label)


def _prepare_context_or_query(
        cat1_indices: list[int],
        cat2_indices: list[int],
        cat1_spurious_labels: list[int],
        cat2_spurious_labels: list[int],
        cat1_class_label: int,
        cat2_class_label: int) -> list[Example]:
    """Combines and shuffles list of examples from 2 classes."""
    cat1_class_labels = [cat1_class_label] * len(cat1_indices)
    cat2_class_labels = [cat2_class_label] * len(cat2_indices)
    cat1_examples = list(zip(
        cat1_indices, cat1_spurious_labels, cat1_class_labels))
    cat2_examples = list(zip(
        cat2_indices, cat2_spurious_labels, cat2_class_labels))
    examples = cat1_examples + cat2_examples
    random.shuffle(examples)
    return examples


class INaturalistEmbContextsDatasetV2(Dataset):
    """A dataset class for the iNaturalist dataset, which handles embeddings and contextual data.

    This class supports either generating new data dynamically or loading pre-existing data from a specified path.
    """

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
        saved_data_path (str or None): Path for loading data; if None, new data is generated.
        """
        super(INaturalistEmbContextsDatasetV2, self).__init__()

        # Prepare encodings and data files
        encodings_data = np.load(os.path.join(dataset_path, encoding_extractor, "combined.npz"))
        self._encodings = encodings_data["encodings"]
        self._encodings_indices_map = encodings_data["indices_map"]

        self._dataframe = pd.read_csv(os.path.join(dataset_path, "prepared_data.csv"))
        self._dataframe = self._dataframe[self._dataframe['split'].isin([class1_split, class2_split])]

        self._data_length = data_length

        # Dataset parameters
        self._context_class_size = context_class_size
        self._minority_group_proportion = minority_group_proportion
        self._spurious_setting = spurious_setting

        # Unique categories in the dataset
        self._class1_split = class1_split
        self._class2_split = class2_split
        if class1_split == class2_split:
            self._categories = self._dataframe["category"].unique()
        else:
            self._categories = (self._dataframe.loc[self._dataframe['split'] == class1_split, "category"].unique(),
                                self._dataframe.loc[self._dataframe['split'] == class2_split, "category"].unique())

        # Loading tokens data
        tokens_data = np.load(os.path.join(dataset_path, "avg_norms", f"{encoding_extractor}_l2.npz"), mmap_mode="r")
        # Convert tokens_data to a dictionary to resolve "Bad CRC-32" error in multi-worker mode.
        tokens_data = {k: tokens_data[k] for k in tokens_data.keys()}

        tokens_generator = TokenGenerator(tokens_data=tokens_data,
                                          are_spurious_tokens_fixed=False,
                                          are_class_tokens_fixed=True,
                                          token_generation_mode='opposite')

        self._spurious_tokens_generator, self._class_tokens_generator = tokens_generator()

        self._v1_behavior = v1_behavior

        if rotate_encodings:
            self._img_encoding_transform = EncodingRotator(n_rotation_matrices, tokens_data["token_len"].item())
        else:
            self._img_encoding_transform = IdentityTransform()

        self._saved_data_path = saved_data_path

    def __getitem__(self, idx) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Returns a dataset example given the example index.

        If 'self._saved_data_path' is None, it generates a new data item.
        If 'self._saved_data_path' is not None, it loads the data item from the saved path.

        Args:
            idx (int): The index of the item to retrieve.

        Returns: A tuple (input_seq, context, queries, query_indices), where input_seq is a np.ndarray of shape
            (total_seq_len, dim) that is to be fed ta a V2 transformer. Context and queries are np.ndarrays of shape
            (2 * context_class_size, 3) describing context/query examples with (id, spurious, class) tuples.
            query_indices specifies positions of input_seq corresponding to query tokens.
        """
        if self._saved_data_path is None:
            spurious_tokens = next(self._spurious_tokens_generator)
            class_tokens = next(self._class_tokens_generator)
            context, queries = self._generate_context_and_queries()
        else:
            # TODO(hrayr): need to do something here
            raise NotImplementedError('loading from saved data is not implemented yet.')
            # data = np.load(os.path.join(self._saved_data_path, f"{idx}.npz"))
            # spurious_tokens = data[f"opposite_spurious_tokens"]
            # class_tokens = data[f"opposite_class_tokens"]
            # instance = data["instance"]
            # context_of_ids = [tuple(x) for x in instance[:-1]]
            # query_of_ids = tuple(instance[-1])

        # Process and combine image encodings with tokens
        input_seq, query_indices = self._prepare_transformer_input(
            context=context,
            queries=queries,
            spurious_tokens=spurious_tokens,
            class_tokens=class_tokens)

        if self._v1_behavior:
            queries[:-1] = context[1:]

        return input_seq, np.array(context), np.array(queries), query_indices

    def __len__(self):
        """Returns the total number of ICL instances."""
        return self._data_length

    def _generate_context_and_queries(self) -> (list[Example], list[Example]):
        """
        Constructs a context dataset and a query instance using identifiers and labels
        generated from two randomly selected categories for machine learning.

        This method uses 'self._dataframe' as the source data and utilizes various class attributes
        like 'categories', 'context_class_size', 'minority_group_proportion', etc.

        Returns:
            tuple: A tuple containing the combined context and query examples -- (id, spurious, class) triplets.
        """
        # Randomly selecting two categories
        if self._class1_split != self._class2_split:
            category1, category2 = (np.random.choice(x, size=1)[0] for x in self._categories)
        else:
            category1, category2 = np.random.choice(self._categories, size=2, replace=False)

        # Sampling 2 * _context_class_size examples from each category (half for context and half for queries)
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
        context = _prepare_context_or_query(
            cat1_indices=cat1_context_indices,
            cat2_indices=cat2_context_indices,
            cat1_spurious_labels=cat1_context_spurious_labels,
            cat2_spurious_labels=cat2_context_spurious_labels,
            cat1_class_label=cat1_class_label,
            cat2_class_label=cat2_class_label)

        queries = _prepare_context_or_query(
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

    def _maybe_rotate_embeddings(
            self,
            context_img_encodings: np.ndarray,
            query_img_encodings: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        joint = np.concatenate([context_img_encodings, query_img_encodings],
                               axis=0)
        joint = self._img_encoding_transform(joint)
        n = context_img_encodings.shape[0]
        return joint[:n], joint[n:]

    def _prepare_transformer_input(
            self,
            context: list[Example],
            queries: list[Example],
            spurious_tokens: np.ndarray,
            class_tokens: np.ndarray,
    ) -> (np.ndarray, np.ndarray):
        """Prepares the final input sequence to feed to the transformer given context and queries.

        Args:
            context: List of tuples containing image IDs, spurious labels, and class labels for the context images.
            queries: List of tuples containing image ID, spurious label, and class label for the queries.
            spurious_tokens: numpy array mapping spurious labels to tokens.
            class_tokens: numpy array mapping class labels to tokens.

        Returns: a pair of numpy arrays (input_seq, query_indices), where input_seq has shape
            (total_seq_len, dim) and is to be fed to a V2 ICL transformer, while query_indices
            indicates query positions within the input_seq.
        """
        assert len(context) == len(queries)
        context_img_encodings = self._prepare_image_encodings(context)
        query_img_encodings = self._prepare_image_encodings(queries)

        # rotate embeddings if specified
        context_img_encodings, query_img_encodings = self._maybe_rotate_embeddings(
            context_img_encodings=context_img_encodings,
            query_img_encodings=query_img_encodings)

        input_seq = []
        for i in range(len(context)):
            # Add current context related tokens.
            _, sp, label = context[i]
            input_seq += get_context_example_tokens(
                img_encoding=context_img_encodings[i],
                spurious_token=spurious_tokens[sp],
                class_token=class_tokens[label],
                spurious_setting=self._spurious_setting,
            )

            # Add current query related token.
            # NOTE: no matter what spurious setting we use, query spurious label
            #       and class label will not get their own tokens.
            _, sp, _ = queries[i]
            input_seq += get_query_example_tokens(
                img_encoding=query_img_encodings[i],
                spurious_token=spurious_tokens[sp],
                spurious_setting=self._spurious_setting,
            )

        input_seq = np.stack(input_seq)

        if self._spurious_setting in ['no_spurious', 'sum']:
            query_indices = np.arange(2, input_seq.shape[0], 3)
        elif self._spurious_setting in ['separate_token', 'sum_with_spurious']:
            query_indices = np.arange(3, input_seq.shape[0], 4)
        else:
            raise ValueError(
                f"Invalid spurious setting: '{self._spurious_setting}'. "
                f"Expected 'no_spurious', 'sum', 'separate_token' or 'sum_with_spurious'.")

        return input_seq, query_indices
