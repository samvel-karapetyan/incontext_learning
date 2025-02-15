import dataclasses
import logging
import pandas as pd
import string
import os.path

from torch.utils.data import Dataset
from operator import itemgetter
import numpy as np
import zipfile
import requests
import tempfile
from tqdm import tqdm

# Define a type alias for example data structure
Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)

# Set up logging
log = logging.getLogger(__name__)


class MultiNLISubsetForEncodingExtraction(Dataset):
    """
    A subset of the MultiNLI dataset used for encoding extraction.

    Args:
        sentence_pairs (list of tuple): List of sentence pairs.
        indices (list of int): List of indices corresponding to the dataset split.
    """
    def __init__(self, sentence_pairs, indices):
        self.sentence_pairs = sentence_pairs
        self.indices = indices

    def __getitem__(self, idx):
        """Retrieve a sentence pair and its corresponding index."""
        x = self.sentence_pairs[idx]
        return x, self.indices[idx]

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.sentence_pairs)


class MultiNLIForEncodingExtraction(Dataset):
    """
    A custom dataset class for handling the MultiNLI dataset, specifically designed for encoding extraction.

    Args:
        root_dir (str): Directory where the dataset is located.
        download (bool): Whether to download the dataset if it does not exist locally.
    """
    def __init__(self, root_dir, download=False):
        super().__init__()

        # Path to the dataset directory
        self.dataset_path = os.path.join(root_dir, "multinli_1.0")
        
        # Check if the dataset exists, download if necessary
        if not os.path.exists(self.dataset_path):
            if download:
                self.download_dataset(root_dir)
            else:
                raise FileNotFoundError("Dataset not found.")

        # Path to the prepared data file
        prepared_data_path = os.path.join(self.dataset_path, "prepared_data.csv")

        # Prepare data if not already done
        if not os.path.exists(prepared_data_path):
            self.prepare_data_and_save(prepared_data_path)

        # Load prepared data into a pandas DataFrame
        self.dataset = pd.read_csv(prepared_data_path)

        # Create a list of sentence pairs
        self.sentence_pairs = list(zip(self.dataset['sentence1'], self.dataset['sentence2']))

    def get_subset(self, split):
        """
        Retrieve a subset of the dataset based on the specified split (train/val/test).

        Args:
            split (str): One of 'train', 'val', or 'test'.

        Returns:
            MultiNLISubsetForEncodingExtraction: Subset of the dataset.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Unexpected value {split=}")

        split_indices = np.where(self.dataset['split'] == split)[0]

        return MultiNLISubsetForEncodingExtraction(
            sentence_pairs=itemgetter(*split_indices)(self.sentence_pairs),
            indices=split_indices
        )

    def prepare_data_and_save(self, prepared_data_path):
        """
        Prepare the MultiNLI dataset and save it as a CSV file.

        Args:
            prepared_data_path (str): Path to save the prepared data file.
        """
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

                # Assign a portion of training data to validation split
                rng = np.random.default_rng(seed=42)
                val_count = int(0.2 * len(df))
                val_indices = rng.choice(np.arange(len(df)), val_count)

                split_array[val_indices] = "val"
                df['split'] = split_array
            else:
                df['split'] = ["test"] * len(df)

            dataset = pd.concat([dataset, df[columns_to_collect]], ignore_index=True)

        # Remove unlabeled and invalid data
        dataset = dataset[dataset['gold_label'] != '-']
        dataset = dataset[dataset['sentence2'] != 'n/a']

        # Add additional features
        dataset['sentence2_has_negation'] = dataset['sentence2'].apply(self._is_sentence_has_negation)
        dataset['y_array'] = dataset['gold_label'].map(labels_map)

        # Save the prepared dataset to CSV
        dataset[['sentence1', 'sentence2', 'y_array', 'sentence2_has_negation', 'split']].to_csv(prepared_data_path, index=False)

    def download_dataset(self, root_dir):
        """
        Download the MultiNLI dataset and extract it.

        Args:
            root_dir (str): Directory to save the downloaded dataset.
        """
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
        """
        Check if a sentence contains negation words.

        Args:
            sentence (str): Input sentence.

        Returns:
            int: 1 if the sentence contains negation words, else 0.
        """
        def tokenize(s):
            # Remove punctuation and convert to lowercase
            s = s.translate(str.maketrans('', '', string.punctuation))
            s = s.lower()
            s = s.split(' ')
            return s

        negation_words = ['nobody', 'no', 'never', 'nothing']

        return int(any(negation_word in tokenize(sentence) for negation_word in negation_words))


@dataclasses.dataclass
class CustomExtractedMultiNLISubset:
    """
    A data structure for holding extracted encodings and associated labels for the MultiNLI dataset.

    Attributes:
        encodings (np.ndarray): Encodings of the dataset.
        y_array (np.ndarray): Labels of the dataset.
        c_array (np.ndarray): Spurious labels of the dataset.
    """
    encodings: np.ndarray
    y_array: np.ndarray
    c_array: np.ndarray

    def __len__(self):
        return len(self.y_array)


class MultiNLISubsetExtracted(Dataset):
    """
    A subset of extracted encodings from the MultiNLI dataset.

    Args:
        ds (CustomExtractedMultiNLISubset): Extracted subset data.
        reverse_task (bool): Whether to reverse the task (swap y and c).
    """
    def __init__(self, ds: CustomExtractedMultiNLISubset, reverse_task: bool = False):
        self.ds = ds
        self._reverse_task = reverse_task

    def __getitem__(self, indices) -> tuple[np.ndarray, Examples]:
        """Retrieve encodings and examples by indices."""
        x = self.ds.encodings[indices].copy()
        y = self.ds.y_array[indices]
        c = self.ds.c_array[indices]

        # Create examples based on reverse task setting
        if not self._reverse_task:
            examples = np.stack([indices, c, y], axis=1)
        else:
            examples = np.stack([indices, y, c], axis=1)

        return x, examples

    def __len__(self):
        return len(self.ds)


class MultiNLIExtracted:
    """
    Handles loading and processing of extracted encodings for the MultiNLI dataset.

    Args:
        dataset_path (str): Path to the dataset directory.
        encoding_extractor (str): Encoding extractor type.
        reverse_task (bool): Whether to reverse the task.
    """
    def __init__(self, dataset_path: str, encoding_extractor: str, reverse_task: bool = False):
        self._dataset_path = dataset_path
        self._encoding_extractor = encoding_extractor
        self._reverse_task = reverse_task

        # Load labels and spurious labels
        prepared_data = pd.read_csv(os.path.join(dataset_path, 'multinli_1.0', 'prepared_data.csv'))
        self.y_array = prepared_data['y_array'].to_numpy()
        self.c_array = prepared_data['sentence2_has_negation'].to_numpy()

    def get_subset(self, split, *args, **kwargs) -> MultiNLISubsetExtracted:
        """
        Retrieve a subset of the dataset based on the specified split.

        Args:
            split (str): One of 'train', 'val', or 'test'.

        Returns:
            MultiNLISubsetExtracted: Subset of the dataset.
        """
        assert split in ['train', 'val', 'test']

        encodings_path = os.path.join(self._dataset_path, "multinli", self._encoding_extractor,
                                      split, "combined.npz")

        # Load encodings and indices
        encodings, indices_map = np.load(encodings_path).values()

        split_indices = np.where(indices_map != -1)[0]

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
        )
