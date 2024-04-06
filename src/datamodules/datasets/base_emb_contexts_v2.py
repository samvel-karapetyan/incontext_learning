from abc import ABC, abstractmethod
from typing import Optional
import logging
import os
import numpy as np

from torch.utils.data import Dataset
from src.utils.dataset_helpers import TokenGenerator
from src.utils.dataset_helpers.encoding_transforms import EncodingRotator, IdentityTransform
from src.utils.dataset_helpers.context_prep_utils import get_context_example_tokens,\
    get_query_example_tokens

log = logging.getLogger(__name__)

Example = tuple[int, int, int]  # (index, spurious_label, class_label)


class BaseEmbContextsDatasetV2(Dataset, ABC):
    """Base class for V2 datasets."""

    def __init__(self,
                 encoding_extractor: str,
                 data_length: int,
                 context_class_size: int,
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
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_class_size (int): The size of each class in the context.
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
        super(BaseEmbContextsDatasetV2, self).__init__()

        self._encoding_extractor = encoding_extractor
        self._data_length = data_length
        self._context_class_size = context_class_size
        self._spurious_setting = spurious_setting
        self._v1_behavior = v1_behavior
        self._rotate_encodings = rotate_encodings
        self._n_rotation_matrices = n_rotation_matrices
        self._label_noise_ratio_interval = label_noise_ratio_interval
        self._input_noise_std_interval = input_noise_std_interval
        self._permute_input_dim = permute_input_dim
        self._ask_context_prob = ask_context_prob
        self._saved_data_path = saved_data_path

        # Loading tokens data
        token_data_path = os.path.join(
            '/nfs/np/mnt/xtb/rafayel/data/inaturalist2017/avg_norms',  # TODO(hrayr): fix data path hard coding
            f"{encoding_extractor}_l2.npz")
        tokens_data = np.load(token_data_path, mmap_mode="r")
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
        if self._saved_data_path is not None:
            # TODO(hrayr): implement loading from a cached dataset
            raise NotImplementedError('loading from saved data is not implemented yet.')

        # get spurious and class tokens
        spurious_tokens = next(self._spurious_tokens_generator)
        class_tokens = next(self._class_tokens_generator)

        # generate context and query examples
        context, queries = self._generate_context_and_queries()
        assert len(context) == len(queries)

        # Deciding which queries should be replaced with which context examples.
        # This depends on v1 behavior and self.ask_context_prob.
        # Setting -1 indicates keeping the original query.
        query_to_context_map = -np.ones(len(context), dtype=np.int32)  # keeping all original queries

        if self._ask_context_prob is not None:
            for q_idx in range(len(context)):
                if np.random.rand() < self._ask_context_prob:
                    c_idx = np.random.randint(q_idx + 1)  # selecting one of previous context examples
                    query_to_context_map[q_idx] = c_idx

        # When v1 behavior is enabled, we need to replace the remaining original queries
        # with corresponding context examples.
        if self._v1_behavior:
            for q_idx in range(len(context) - 1):  # excluding the last query
                if query_to_context_map[q_idx] == -1:
                    query_to_context_map[q_idx] = q_idx + 1  # next unseen context example

        # construct context and query image encodings
        # update `queries`
        context_img_encodings = self._prepare_context_image_encodings(context)
        query_img_encodings = self._prepare_query_image_encodings(queries)
        for q_idx in range(len(context)):
            c_idx = query_to_context_map[q_idx]
            if c_idx != -1:
                query_img_encodings[q_idx] = context_img_encodings[c_idx]
                queries[q_idx] = context[c_idx]

        # do data augmentations if needed
        context_img_encodings, query_img_encodings = self._maybe_rotate_embeddings(
            context_img_encodings=context_img_encodings,
            query_img_encodings=query_img_encodings)

        context_img_encodings, query_img_encodings = self._maybe_permute_embeddings(
            context_img_encodings=context_img_encodings,
            query_img_encodings=query_img_encodings
        )

        self._maybe_add_label_noise(context)

        context_img_encodings = self._maybe_add_input_noise(
            context_img_encodings=context_img_encodings
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

        return input_seq, np.array(context), np.array(queries), query_indices

    def __len__(self):
        """Returns the total number of ICL instances."""
        return self._data_length

    @abstractmethod
    def _generate_context_and_queries(self) -> (list[Example], list[Example]):
        """Should sample context and query examples; and return (context, queries) pair."""
        pass

    @abstractmethod
    def _prepare_context_image_encodings(
            self,
            context: list[Example]
    ) -> np.ndarray:
        """Should return a np.ndarray of stacked embeddings of context examples."""
        pass

    @abstractmethod
    def _prepare_query_image_encodings(
            self,
            queries: list[Example]
    ) -> np.ndarray:
        """Should return a np.ndarray of stacked embeddings of queries."""
        pass

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
    
    def _maybe_permute_embeddings(
            self,
            context_img_encodings: np.ndarray,
            query_img_encodings: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        if self._permute_input_dim:
            random_permutation = np.random.permutation(context_img_encodings.shape[1])

            context_img_encodings = context_img_encodings[:, random_permutation]
            query_img_encodings = query_img_encodings[:, random_permutation]

        return context_img_encodings, query_img_encodings

    def _maybe_add_label_noise(
            self,
            context: list[Example],
    ) -> None:
        if self._label_noise_ratio_interval:
            low, high = self._label_noise_ratio_interval

            flip_p = np.random.uniform(low=low, high=high)
            label_noise_count = int(flip_p * len(context))
            label_noise_indices = np.random.choice(len(context), label_noise_count, replace=False)

            for i in label_noise_indices:
                idx, sp, label = context[i]
                context[i] = (idx, sp, 1 - label)

    def _maybe_add_input_noise(
            self,
            context_img_encodings: np.ndarray
    ) -> np.ndarray:
        if self._input_noise_std_interval:
            low, high = self._input_noise_std_interval

            input_noise_std = np.random.uniform(low=low, high=high)
            input_noise = np.random.normal(
                scale=input_noise_std, size=context_img_encodings.shape,
            ).astype(context_img_encodings.dtype)

            context_img_encodings = context_img_encodings + input_noise

        return context_img_encodings
