from typing import Optional
import logging
import os
import random

from torch.utils.data import Dataset
import numpy as np

from src.utils.dataset_helpers import TokenGenerator
from src.datamodules.datasets.waterbirds import CustomizedWaterbirdsDataset as WaterbirdsDataset
from src.utils.dataset_helpers.context_prep_utils import get_group_counts_based_on_proportions,\
    get_context_example_tokens, get_query_example_tokens

log = logging.getLogger(__name__)

Example = tuple[int, int, int]  # (index, spurious_label, class_label)


class WaterbirdsEmbContextsDatasetV2(Dataset):
    """
    A dataset class for the Waterbirds dataset, which handles embeddings and contextual data.
    This class supports either generating new data dynamically or loading pre-existing data from a specified path.
    """

    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 data_length: int,
                 context_class_size: int,
                 group_proportions: list[float],
                 spurious_setting: str,
                 v1_behavior: bool = False,
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
        saved_data_path (str or None): Path for loading data; if None, new data is generated.
        """
        super(WaterbirdsEmbContextsDatasetV2, self).__init__()

        dataset = WaterbirdsDataset(root_dir,
                                    data_type="encoding",
                                    encoding_extractor=encoding_extractor,
                                    return_labels=True)
        groups = np.stack([(2 * y + c) for _, y, c, _ in dataset])

        self._train_set = dataset.get_subset("train")
        self._val_set = dataset.get_subset("val")

        self._train_groups = groups[self._train_set.indices]
        self._val_groups = groups[self._val_set.indices]

        self._data_length = data_length

        # Dataset parameters
        self._context_class_size = context_class_size
        self._group_proportions = group_proportions

        # Loading tokens data
        tokens_data = np.load(os.path.join(root_dir, "inaturalist2017", "avg_norms", f"{encoding_extractor}_l2.npz"),
                              mmap_mode="r")
        # Convert tokens_data to a dictionary to resolve "Bad CRC-32" error in multi-worker mode.
        tokens_data = {k: tokens_data[k] for k in tokens_data.keys()}

        tokens_generator = TokenGenerator(tokens_data=tokens_data,
                                          are_spurious_tokens_fixed=False,
                                          are_class_tokens_fixed=True,
                                          token_generation_mode='opposite')

        self._spurious_tokens_generator, self._class_tokens_generator = tokens_generator()

        self._spurious_setting = spurious_setting

        self._v1_behavior = v1_behavior

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
            # TODO(hrayr): do something here
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

    def __len__(self) -> int:
        return self._data_length

    def _generate_context_and_queries(self) -> (list[Example], list[Example]):
        """
        Constructs a context dataset and a query instance using identifiers and labels
        generated from two randomly selected categories for machine learning.

        Returns:
            tuple: A tuple containing the combined context dataset and the query instance, both primarily based on IDs.
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

        context_img_encodings = np.stack(
            [self._train_set[idx][0] for idx, _, _ in context]
        )
        query_img_encodings = np.stack(
            [self._val_set[idx][0] for idx, _, _ in queries]
        )

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
