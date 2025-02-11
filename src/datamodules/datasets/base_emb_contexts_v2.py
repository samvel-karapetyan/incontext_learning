from abc import ABC, abstractmethod
from typing import Optional
import logging
import os

import numpy as np

from torch.utils.data import Dataset
from src.utils.dataset_helpers import TokenGenerator, EncodingRotator, IdentityTransform, PartlySwapper
from src.utils.dataset_helpers.context_prep_utils import get_context_example_tokens, \
    get_query_example_token

log = logging.getLogger(__name__)

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)


class BaseEmbContextsDatasetV2(Dataset, ABC):
    """Base class for V2 datasets."""

    def __init__(self,
                 encoding_extractor: str,
                 data_length: int,
                 context_class_size: int,
                 rotate_encodings: bool = False,
                 n_rotation_matrices: Optional[int] = None,
                 permute_input_dim: bool = False,
                 ):
        """
        Arguments:
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_class_size (int): The size of each class in the context.
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
        permute_input_dim (bool): Determines if image encodings are permuted. 
                                True enables permutation, while False bypasses it.
        """
        super(BaseEmbContextsDatasetV2, self).__init__()

        self._encoding_extractor = encoding_extractor
        self._data_length = data_length
        self._context_class_size = context_class_size
        self._rotate_encodings = rotate_encodings
        self._n_rotation_matrices = n_rotation_matrices
        self._permute_input_dim = permute_input_dim

        # Loading tokens data
        if self._encoding_extractor in ["stroberta", "bert", "roberta"]:
            token_data_path = os.path.join(
                os.environ.get('DATA_ROOT_DIR'),
                'multinli/avg_norms',
                f"{encoding_extractor}_l2.npz")
        else:
            token_data_path = os.path.join(
                os.environ.get('DATA_ROOT_DIR'),
                'inaturalist2017/avg_norms',
                f"{encoding_extractor}_l2.npz")


        tokens_data = np.load(token_data_path)
        # Convert tokens_data to a dictionary to resolve "Bad CRC-32" error in multi-worker mode.
        self._tokens_data = {k: tokens_data[k] for k in tokens_data.keys()}

        tokens_generator = TokenGenerator(tokens_data=self._tokens_data)

        (self._query_token_generator,
         self._class_tokens_generator) = tokens_generator()

        if rotate_encodings:
            self._img_encoding_transform = EncodingRotator(n_rotation_matrices, tokens_data["token_len"].item())
        else:
            self._img_encoding_transform = IdentityTransform()


    def __getitem__(self, idx) -> (np.ndarray, Examples, Examples, np.ndarray):
        """Returns a dataset example given the example index.

        Args:
            idx (int): The index of the item to retrieve.
        """

        query_token = self._query_token_generator()
        class_tokens = self._class_tokens_generator()

        context, queries = self._generate_context_and_queries(
            num_context_examples=2 * self._context_class_size,
            num_query_examples=1,
        )

        context_img_encodings = self._prepare_context_image_encodings(context)
        query_img_encoding = self._prepare_query_image_encodings(queries)

        # do data augmentations if needed
        context_img_encodings, query_img_encoding = self._maybe_rotate_embeddings(
            context_img_encodings=context_img_encodings,
            query_img_encodings=query_img_encoding)
        
        context_img_encodings, query_img_encoding = self._maybe_permute_embeddings(
            context_img_encodings=context_img_encodings,
            query_img_encodings=query_img_encoding
        )

        input_seq = [get_query_example_token(img_encoding=query_img_encoding[0])[0]]
        query_indices = []
        for i in range(len(context)):
            # Add current context related tokens.
            _, _, label = context[i]
            input_seq.extend(get_context_example_tokens(
                img_encoding=context_img_encodings[i],
                class_token=class_tokens[label],
            ))

            query_indices.append(len(input_seq))
            input_seq.extend(query_token)

        input_seq = np.stack(input_seq)
        queries = np.tile(queries, (context.shape[0], 1))

        return input_seq, context, queries, np.array(query_indices)

    def __len__(self):
        """Returns the total number of ICL instances."""
        return self._data_length

    @abstractmethod
    def _generate_context_and_queries(
            self,
            num_context_examples: int,
            num_query_examples: int,
    ) -> (Examples, Examples):
        """Should sample context and query examples with configured group proportions."""
        pass

    @abstractmethod
    def _prepare_context_image_encodings(
            self,
            context: Examples,
    ) -> np.ndarray:
        """Should return a np.ndarray of stacked embeddings of context examples."""
        pass

    @abstractmethod
    def _prepare_query_image_encodings(
            self,
            queries: Examples,
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
