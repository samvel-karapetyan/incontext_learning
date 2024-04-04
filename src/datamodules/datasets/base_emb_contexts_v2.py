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

        # Switch to v1 behavior if needed
        if self._v1_behavior:
            queries[:-1] = context[1:]

        # Process and combine image encodings with tokens
        input_seq, query_indices = self._prepare_transformer_input(
            context=context,
            queries=queries,
            spurious_tokens=spurious_tokens,
            class_tokens=class_tokens)

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
        context_img_encodings = self._prepare_context_image_encodings(context)
        if not self._v1_behavior:
            query_img_encodings = self._prepare_query_image_encodings(queries)
        else:
            # only the last query is from the query distribution, other are actually the context examples
            query_img_encodings = np.concatenate([
                self._prepare_context_image_encodings(queries[:-1]),
                self._prepare_query_image_encodings(queries[-1:]),
            ], axis=0)

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
