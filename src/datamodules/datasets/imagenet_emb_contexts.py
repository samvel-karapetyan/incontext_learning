import logging
import os

import pandas as pd
import numpy as np

from src.datamodules.datasets import ImagenetDataset
from src.utils.dataset_helpers import TokenGenerator
from src.datamodules.datasets.base_emb import BaseEmbContextsDataset

log = logging.getLogger(__name__)


class ImagenetEmbContextsDataset(BaseEmbContextsDataset):
    """
    A dataset class for the ImageNet dataset, which handles embeddings and contextual data.
    This class supports either generating new data dynamically or loading pre-existing data from a specified path.

    Attributes:
        dataset_path (str): The path to the dataset directory.
        encoding_extractor (str): The name of the encoding extractor used.
        data_length (int): The length of the dataset.
        context_class_size (int): The size of each class in the context.
        minority_group_proportion (float): The proportion of the minority group in the context per class.
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
                 dataset_path,
                 encoding_extractor,
                 data_length,
                 context_class_size,
                 minority_group_proportion,
                 are_spurious_tokens_fixed,
                 are_class_tokens_fixed,
                 token_generation_mode,
                 spurious_setting,
                 saved_data_path=None):
        super(ImagenetEmbContextsDataset, self).__init__(
            dataset_path=dataset_path,
            encoding_extractor=encoding_extractor,
            data_length=data_length,
            context_class_size=context_class_size,
            minority_group_proportion=minority_group_proportion,
            token_generation_mode=token_generation_mode,
            spurious_setting=spurious_setting,
            saved_data_path=saved_data_path,
        )

        self._dataframe = pd.read_csv(os.path.join(dataset_path, ImagenetDataset.annotation_files['train']), sep=' ',
                                      header=None, names=['filepath', 'id'])

        self._dataframe["category"] = self._dataframe["filepath"].apply(lambda x: x.split('/')[0])

        # Calculate the value counts of categories that appear more than context_class_size times
        valid_categories = self._dataframe['category'].value_counts()[lambda x: x > context_class_size].index

        # Filter the dataframe to only include rows with those categories
        self._dataframe = self._dataframe[self._dataframe['category'].isin(valid_categories)]

        self._categories = self._dataframe["category"].unique()

        # Loading tokens data
        tokens_data = np.load(os.path.join("/nfs/np/mnt/xtb/rafayel/data", "inaturalist2017",
                                           "avg_norms", f"{encoding_extractor}_l2.npz"), mmap_mode="r")
        # Convert tokens_data to a dictionary to resolve "Bad CRC-32" error in multi-worker mode.
        tokens_data = {k: tokens_data[k] for k in tokens_data.keys()}

        tokens_generator = TokenGenerator(tokens_data=tokens_data,
                                          are_spurious_tokens_fixed=are_spurious_tokens_fixed,
                                          are_class_tokens_fixed=are_class_tokens_fixed,
                                          token_generation_mode=token_generation_mode)

        self._spurious_tokens_generator, self._class_tokens_generator = tokens_generator()

    def _sample_k_examples(self, category: str, k: int) -> list[int]:
        # Using the class's dataframe
        df = self._dataframe
        return df.loc[df["category"] == category, "id"].sample(k).tolist()

    @property
    def categories(self):
        return self._categories

    @property
    def spurious_tokens_generator(self):
        return self._spurious_tokens_generator

    @property
    def class_tokens_generator(self):
        return self._class_tokens_generator
