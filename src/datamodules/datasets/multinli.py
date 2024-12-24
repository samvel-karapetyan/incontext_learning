from typing import Optional
import dataclasses
import logging
import pandas as pd
import string
import os.path

from torch.utils.data import Dataset
from operator import itemgetter
import numpy as np
from typing import Tuple

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)

log = logging.getLogger(__name__)

import zipfile
import requests
import tempfile

from tqdm import tqdm


class MultiNLISubsetForEncodingExtraction(Dataset):
    def __init__(self, sentence_pairs, indices):
        self.sentence_pairs = sentence_pairs
        self.indices = indices

    def __getitem__(self, idx):
        x = self.sentence_pairs[idx]
        return x, self.indices[idx]

    def __len__(self):
        return len(self.sentence_pairs)


class MultiNLIForEncodingExtraction(Dataset):
    """
    A custom dataset class for the MultiNLI dataset.

    Args:
        root_dir (str): The file path to the dataset directory.
    """

    def __init__(self, root_dir, download=False):
        super().__init__()

        self.dataset_path = os.path.join(root_dir, "multinli_1.0")
        
        if not os.path.exists(self.dataset_path):
            if download:
                self.download_dataset(root_dir)
            else:
                raise("Dataset not found.")

        prepared_data_path = os.path.join(self.dataset_path, "prepared_data.csv")

        if not os.path.exists(prepared_data_path):
            self.prepare_data_and_save(prepared_data_path)

        self.dataset = pd.read_csv(prepared_data_path)

        self.sentence_pairs = list(zip(self.dataset['sentence1'], self.dataset['sentence2']))

    def get_subset(self, split):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Unexpected value {split=}")      

        split_indices = np.where(self.dataset['split'] == split)[0]

        return MultiNLISubsetForEncodingExtraction(
            sentence_pairs=itemgetter(*split_indices)(self.sentence_pairs),
            indices=split_indices
        )

    def prepare_data_and_save(self, prepared_data_path):
        data_files = [
            "multinli_1.0_train.jsonl", 
            "multinli_1.0_dev_matched.jsonl", 
        ]

        labels_map = {
            'neutral': 0, 
            'entailment': 0, 
            'contradiction': 1
        }

        columns_to_collect = ['sentence1', 'sentence2', 'gold_label', 'split']
        dataset = pd.DataFrame()
        
        for file_name in data_files:
            file_path = os.path.join(self.dataset_path, file_name)

            df = pd.read_json(path_or_buf=file_path, lines=True)

            if "train" in file_name:
                split_array = np.array(["train"] * len(df))

                rng = np.random.default_rng(seed=42)
                val_count = int(0.2 * len(df))
                val_indices = rng.choice(np.arange(len(df)), val_count)

                split_array[val_indices] = "val"

                df['split'] = split_array
            else:
                df['split'] = ["test"] * len(df)

            dataset = pd.concat([dataset, df[columns_to_collect]], ignore_index=True)

        
        dataset = dataset[dataset['gold_label'] != '-'] # remove unlabeled data
        dataset = dataset[dataset['sentence2'] != 'n/a'] # remove n/a

        dataset['sentence2_has_negation'] = dataset['sentence2'].apply(self._is_sentence_has_negation)
        dataset['y_array'] = dataset['gold_label'].map(labels_map)

        dataset[['sentence1', 'sentence2', 'y_array', 'sentence2_has_negation', 'split']].to_csv(prepared_data_path, index=False)

    def download_dataset(self, root_dir):
        url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"

        log.info(f"Downloading the MultiNLI from url: {url}")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        tmp = tempfile.NamedTemporaryFile()

        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            with open(tmp.name, 'wb') as temp_file:
                for chunk in response.iter_content(chunk_size=1024):
                    pbar.update(len(chunk))
                    temp_file.write(chunk)

            with zipfile.ZipFile(tmp.name, 'r') as zip_ref:
                zip_ref.extractall(root_dir)
    
    def _is_sentence_has_negation(self, sentence):
        def tokenize(s): # function inspired by groupDRO tokenization
            s = s.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
            s = s.lower()
            s = s.split(' ')
            return s

        negation_words = ['nobody', 'no', 'never', 'nothing'] # Taken from https://arxiv.org/pdf/1803.02324.pdf

        return int(any(negation_word in tokenize(sentence) for negation_word in negation_words))


@dataclasses.dataclass
class CustomExtractedMultiNLISubset:
    encodings: np.ndarray
    y_array: np.ndarray
    c_array: np.ndarray

    def __len__(self):
        return len(self.y_array)


class MultiNLISubsetExtracted(Dataset):
    def __init__(self,
                 ds: CustomExtractedMultiNLISubset,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self.ds = ds
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add

    def __getitem__(self, indices) -> Tuple[np.ndarray, Examples]:
        x = self.ds.encodings[indices].copy()
        y = self.ds.y_array[indices]
        c = self.ds.c_array[indices]

        # add more negation information if specified
        if self._sp_vector_to_add is not None:
            x += np.outer(2 * c - 1, self._sp_vector_to_add)

        # reverse the task if specified
        if not self._reverse_task:
            examples = np.stack([indices, c, y], axis=1)
        else:
            examples = np.stack([indices, y, c], axis=1)

        return x, examples

    def __len__(self):
        return len(self.ds)


class MultiNLIExtracted:
    def __init__(self,
                 dataset_path: str,
                 encoding_extractor: str,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self._dataset_path = dataset_path
        self._encoding_extractor = encoding_extractor
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add

        prepared_data = pd.read_csv(os.path.join(dataset_path, 'multinli_1.0', 'prepared_data.csv'))
        self.y_array = prepared_data['y_array'].to_numpy()
        self.c_array = prepared_data['sentence2_has_negation'].to_numpy()

    def get_subset(self, split, *args, **kwargs) -> MultiNLISubsetExtracted:
        assert split in ['train', 'val', 'test']

        encodings_path = os.path.join(self._dataset_path, "multinli", self._encoding_extractor, 
                                      split, "combined.npz")

        encodings, indices = np.load(encodings_path).values()

        split_indices = np.where(indices != -1)[0]

        y_array = self.y_array[split_indices]
        c_array = self.c_array[split_indices]

        ds = CustomExtractedMultiNLISubset(
                encodings=encodings,
                y_array=y_array,
                c_array=c_array,
        )

        return MultiNLISubsetExtracted(
                ds=ds,
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add,
            )
