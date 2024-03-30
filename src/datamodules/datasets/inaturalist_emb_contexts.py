import logging
import os

import numpy as np
import pandas as pd

from src.datamodules.datasets.base_emb_contexts import BaseEmbContextsDataset
from src.utils.dataset_helpers import TokenGenerator
from src.utils.dataset_helpers.encoding_transforms import EncodingRotator, IdentityTransform

log = logging.getLogger(__name__)


class INaturalistEmbContextsDataset(BaseEmbContextsDataset):
    """
    A dataset class for the iNaturalist dataset, which handles embeddings and contextual data.
    This class supports either generating new data dynamically or loading pre-existing data from a specified path.

    Attributes:
        dataset_path (str): The path to the dataset directory.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        class1_split (str): The type of data split of class 1 (e.g., 'inner_train', 'inner_val', 'outer').
        class2_split (str): The type of data split of class 2 (e.g., 'inner_train', 'inner_val', 'outer').
        context_class_size (int): The size of each class in the context.
        minority_group_proportion (float): The proportion of the minority group in the context per class.
        are_spurious_tokens_fixed (bool): Flag indicating whether to use fixed spurious tokens.
        are_class_tokens_fixed (bool): Flag indicating whether to use fixed class tokens.
        token_generation_mode (str): Mode of token generation. Accepts 'random' or 'opposite'.
                                     'random' generates tokens with normal distribution,
                                     and 'opposite' generates a pair of tokens where the second is the negative of the first.
        spurious_setting (str): Determines the handling mode of spurious tokens in the dataset instances.
                                Options include 'separate_token'(x,c) , 'no_spurious'(x), 'sum'(x+c)
        rotate_encodings (bool): Determines if image encodings are rotated. True enables rotation
                                 based on class labels, while False bypasses rotation.
        class_dependant_rotate (bool):  Flag decides whether image embedding rotations are class-specific.
        n_rotation_matrices (int): Specifies the number of rotation matrices to generate and store.
        saved_data_path (str or None): Path for loading data; if None, new data is generated.
    """

    def __init__(self,
                 dataset_path,
                 encoding_extractor,
                 data_length,
                 class1_split,
                 class2_split,
                 context_class_size,
                 minority_group_proportion,
                 are_spurious_tokens_fixed,
                 are_class_tokens_fixed,
                 token_generation_mode,
                 spurious_setting,
                 rotate_encodings=False,
                 n_rotation_matrices=None,
                 class_dependant_rotate=None,
                 saved_data_path=None):
        super(INaturalistEmbContextsDataset, self).__init__(
            dataset_path=dataset_path,
            encoding_extractor=encoding_extractor,
            data_length=data_length,
            context_class_size=context_class_size,
            minority_group_proportion=minority_group_proportion,
            token_generation_mode=token_generation_mode,
            spurious_setting=spurious_setting,
            rotate_encodings=rotate_encodings,
            class_dependant_rotate=class_dependant_rotate,
            saved_data_path=saved_data_path)

        self._dataframe = pd.read_csv(os.path.join(dataset_path, "prepared_data.csv"))
        self._dataframe = self._dataframe[self._dataframe['split'].isin([class1_split, class2_split])]

        # Unique categories in the dataset
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
                                          are_spurious_tokens_fixed=are_spurious_tokens_fixed,
                                          are_class_tokens_fixed=are_class_tokens_fixed,
                                          token_generation_mode=token_generation_mode)

        self._spurious_tokens_generator, self._class_tokens_generator = tokens_generator()

        if rotate_encodings:
            self._img_encoding_transform = EncodingRotator(n_rotation_matrices, tokens_data["token_len"].item())
        else:
            self._img_encoding_transform = IdentityTransform()

    def _sample_k_examples(self, category: str, k: int) -> list[int]:
        df = self._dataframe
        return df.loc[df["category"] == category, "image_id"].sample(k).tolist()

    def img_encoding_transform(self, x: np.ndarray) -> np.ndarray:
        return self._img_encoding_transform(x)

    @property
    def categories(self):
        return self._categories

    @property
    def spurious_tokens_generator(self):
        return self._spurious_tokens_generator

    @property
    def class_tokens_generator(self):
        return self._class_tokens_generator
