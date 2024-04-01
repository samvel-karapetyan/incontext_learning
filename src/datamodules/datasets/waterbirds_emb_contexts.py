import logging
import os

from torch.utils.data import Dataset
import numpy as np

from src.utils.dataset_helpers import TokenGenerator
from src.datamodules.datasets.waterbirds import CustomizedWaterbirdsDataset as WaterbirdsDataset
from src.utils.dataset_helpers.context_prep_utils import get_group_counts_based_on_proportions

log = logging.getLogger(__name__)


class WaterbirdsEmbContextsDataset(Dataset):
    """
    A dataset class for the Waterbirds dataset, which handles embeddings and contextual data.
    This class supports either generating new data dynamically or loading pre-existing data from a specified path.

    Attributes:
        root_dir (str): The root directory of the dataset.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_class_size (int): The size of each class in the context.
        group_proportions:
        are_spurious_tokens_fixed (bool): Flag indicating whether to use fixed spurious tokens.
        are_class_tokens_fixed (bool): Flag indicating whether to use fixed class tokens.
        token_generation_mode (str): Mode of token generation. Accepts 'random' or 'opposite'.
                                     'random' generates tokens with normal distribution,
                                     and 'opposite' generates a pair of tokens where the second is the negative of the first.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'separate_token'(x,c) , 'no_spurious'(x), 'sum'(x+c)
        saved_data_path (str or None): Path for loading data; if None, new data is generated.
    """

    def __init__(self,
                 root_dir,
                 encoding_extractor,
                 data_length,
                 context_class_size,
                 group_proportions,
                 are_spurious_tokens_fixed,
                 are_class_tokens_fixed,
                 token_generation_mode,
                 spurious_setting,
                 saved_data_path=None):
        super(WaterbirdsEmbContextsDataset, self).__init__()

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
        tokens_data = np.load(os.path.join(root_dir, "inaturalist2017", "avg_norms", f"{encoding_extractor}_l2.npz"), mmap_mode="r")
        # Convert tokens_data to a dictionary to resolve "Bad CRC-32" error in multi-worker mode.
        tokens_data = {k: tokens_data[k] for k in tokens_data.keys()}

        tokens_generator = TokenGenerator(tokens_data=tokens_data,
                                          are_spurious_tokens_fixed=are_spurious_tokens_fixed,
                                          are_class_tokens_fixed=are_class_tokens_fixed,
                                          token_generation_mode=token_generation_mode)

        self._spurious_tokens_generator, self._class_tokens_generator = tokens_generator()

        self._token_generation_mode = token_generation_mode
        self._saved_data_path = saved_data_path

        self._spurious_setting = spurious_setting

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index. If 'self._saved_data_path' is None,
        it generates a new data item. If 'self._saved_data_path' is not None, it loads the data item from
        the saved path.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
        tuple: A tuple containing the stacked input sequence of image encodings and tokens,
               along with arrays of spurious labels, class labels, and image indices.
        """
        if self._saved_data_path is None:
            spurious_tokens = next(self._spurious_tokens_generator)
            class_tokens = next(self._class_tokens_generator)
            context_of_ids, query_of_ids = self._construct_context_and_query_of_ids()
        else:
            data = np.load(os.path.join(self._saved_data_path, f"{idx}.npz"))
            spurious_tokens = data[f"{self._token_generation_mode}_spurious_tokens"]
            class_tokens = data[f"{self._token_generation_mode}_class_tokens"]
            instance = data["instance"]
            context_of_ids = [tuple(x) for x in instance[:-1]]
            query_of_ids = tuple(instance[-1])

        # Process and combine image encodings with tokens
        input_seq, spurious_labels, class_labels, image_indices = self._process_and_combine_encodings(context_of_ids,
                                                                                                      query_of_ids,
                                                                                                      spurious_tokens,
                                                                                                      class_tokens)
        return input_seq, spurious_labels, class_labels, image_indices

    def __len__(self) -> int:
        return self._data_length

    def _construct_context_and_query_of_ids(self):
        """
        Constructs a context dataset and a query instance using identifiers and labels
        generated from two randomly selected categories for machine learning.

        Returns:
            tuple: A tuple containing the combined context dataset and the query instance, both primarily based on IDs.
        """

        group_counts = get_group_counts_based_on_proportions(
            num_examples=2 * self._context_class_size,
            group_proportions=self._group_proportions)

        context_indices = np.array([], dtype=np.int64)
        for i, group_count in enumerate(group_counts):
            all_group_items = np.where(self._train_groups == i)[0]
            random_items = np.random.choice(all_group_items, group_count, replace=False)
            context_indices = np.concatenate([context_indices, random_items])

        query_group = np.random.randint(0, 4)
        query_idx = np.random.choice(np.where(self._val_groups == query_group)[0])

        np.random.shuffle(context_indices)

        context_labels = np.array([self._train_set[idx][1] for idx in context_indices])
        context_spurious_labels = np.array([self._train_set[idx][2] for idx in context_indices])
        query_label = self._val_set[query_idx][1]
        query_spurious_label = self._val_set[query_idx][2]

        context = list(zip(context_indices, context_spurious_labels, context_labels))
        query = (query_idx, query_spurious_label, query_label)

        return context, query

    def _process_and_combine_encodings(self, context_of_ids, query_of_ids, spurious_tokens, class_tokens):
        """
        Processes and combines image encodings with spurious and class tokens for the given context and query IDs.

        Args:
            context_of_ids (list): List of tuples containing image IDs, spurious labels, and class labels for the context images.
            query_of_ids (tuple): Tuple containing image ID, spurious label, and class label for the query image.
            spurious_tokens (dict): Dictionary mapping spurious labels to tokens.
            class_tokens (dict): Dictionary mapping class labels to tokens.

        Returns:
            tuple: A tuple containing the stacked input sequence of image encodings and tokens,
                   along with arrays of spurious labels, class labels, and image indices.
        """
        input_seq = []
        image_indices = []
        spurious_labels = []
        class_labels = []

        for image_id, spurious_label, class_label in (context_of_ids + [query_of_ids]):
            image_indices.append(image_id)
            spurious_labels.append(spurious_label)
            class_labels.append(class_label)

            image_enc, *_ = self._val_set[image_id] if image_id == query_of_ids[0] else self._train_set[image_id]
            class_token = class_tokens[class_label]

            if self._spurious_setting == 'separate_token':
                spurious_token = spurious_tokens[spurious_label]
                input_seq += [image_enc, spurious_token, class_token]
            elif self._spurious_setting == 'sum':
                spurious_token = spurious_tokens[spurious_label]
                input_seq += [image_enc + spurious_token, class_token]
            elif self._spurious_setting == 'no_spurious':
                input_seq += [image_enc, class_token]
            else:
                raise ValueError(f"Invalid spurious setting: '{self._spurious_setting}'. Expected 'separate_token', 'sum', or 'no_spurious'.")

        input_seq.pop()  # removing the label of query from input sequence

        return np.stack(input_seq), np.array(spurious_labels), np.array(class_labels), np.array(image_indices)