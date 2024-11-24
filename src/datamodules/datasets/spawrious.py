from typing import Optional
import dataclasses
import logging
import os.path

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

Examples = np.ndarray  # shaped (num_examples, 3) with each row being a triplet (index, spurious_label, class_label)

log = logging.getLogger(__name__)


class SpawriousForEncodingExtraction(Dataset):
    """
    A custom dataset class for the Spawrious dataset.

    Args:
        dataset_path (str): The file path to the dataset directory.
    """

    def __init__(
        self,
        dataset_path: str,
        resize_size: int
    ):
        self._dataset_path = dataset_path

        # Check that the dataset exists
        if not os.path.exists(dataset_path):
            raise ValueError(
                "Dataset path does not exist. Please download and unzip the entire Spawrious dataset.")

        self._transform = transforms.Compose([
            transforms.Resize(resize_size),  # Resize the smaller edge to {resize_size} while maintaining aspect ratio
            transforms.ToTensor(),  # Convert to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Normalize with ImageNet mean and std
        ])

        # Enumerate all images; store backgrounds and labels.
        subdirs = ['0', '1']
        backgrounds = ['beach', 'desert', 'dirt', 'jungle', 'mountain', 'snow']
        breeds = ['bulldog', 'corgi', 'dachshund', 'labrador']

        self.full_file_paths = []
        self.bg_array = []
        self.y_array = []

        for subdir in subdirs:
            for background_idx, background in enumerate(backgrounds):
                for breed_idx, breed in enumerate(breeds):
                    dir_path = os.path.join(self._dataset_path, subdir, background, breed)
                    files = sorted(os.listdir(dir_path))
                    for f in files:
                        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):
                            self.full_file_paths.append(os.path.join(dir_path, f))
                            self.bg_array.append(background_idx)
                            self.y_array.append(breed_idx)

        log.info(f'Found {len(self.y_array)} Spawrious images')

        self.full_file_paths = np.array(self.full_file_paths)
        self.bg_array = np.array(self.bg_array)
        self.y_array = np.array(self.y_array)

        np.savez(
            os.path.join(self._dataset_path, 'metadata.npz'),
            bg_array=self.bg_array,
            y_array=self.y_array,
        )

    def __getitem__(self, idx):
        """
        Retrieves an image and its ID at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its corresponding image ID.
        """
        image_path = self.full_file_paths[idx]

        # Load the image from the file system
        image = Image.open(image_path).convert('RGB')

        # Apply the predefined transformations
        image = self._transform(image)

        # Return the transformed image and its ID
        return image, idx

    def __len__(self):
        return len(self.y_array)


@dataclasses.dataclass
class CustomExtractedSpawriousSubset:
    encodings: np.ndarray
    y_array: np.ndarray
    bg_array: np.ndarray

    def __len__(self):
        return len(self.y_array)


class SpawriousSubsetExtracted(Dataset):
    def __init__(self,
                 ds: CustomExtractedSpawriousSubset,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self.ds = ds
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add

    def __getitem__(self, indices) -> (np.ndarray, Examples):
        x = self.ds.encodings[indices].copy()

        # Covert labels to binary. In all our splits, we have 2 tasks:
        # (corgi vs dachshund) and (bulldog vs labrador). For this reason,
        # we assign corgi and bulldog to class 0; and dachshund and labrador to class 1.
        y = self.ds.y_array[indices]
        y = (y >= 2).astype(np.int32)

        background_id = self.ds.bg_array[indices]
        # In our constructed train and val splits, we have mountain (4) vs snow (5)
        # In our constructed test split, we have beach (0) vs desert (1)
        # Therefore, we map backgrounds 0 and 4 to spurious=0 and backgrounds 1 and 5 to spurious=1.
        c = ((background_id == 1) | (background_id == 5)).astype(np.int32)

        # add more background information if specified
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


def _put_encodings_in_right_order(
        indices_within_full_spawrious: np.ndarray,
        encodings: np.ndarray,
        index_map: np.ndarray,
) -> np.ndarray:
    """Permutes rows of `encodings` such that the i-th row corresponds to the i-th example of the subset."""
    n = len(indices_within_full_spawrious)
    row_indices = np.zeros(n, dtype=np.int32)
    for idx in range(n):
        idx_within_full_spawrious = indices_within_full_spawrious[idx],
        encoding_row_index = index_map[idx_within_full_spawrious]
        assert encoding_row_index != -1
        row_indices[idx] = encoding_row_index
    return encodings[row_indices]


class SpawriousExtracted:
    def __init__(self,
                 root_dir: str,
                 encoding_extractor: str,
                 reverse_task: bool = False,
                 sp_vector_to_add: Optional[np.ndarray] = None):
        self._root_dir = root_dir
        self._encoding_extractor = encoding_extractor
        self._reverse_task = reverse_task
        self._sp_vector_to_add = sp_vector_to_add

        # Prepare a simple version of the full spawrious dataset with image embeddings
        metadata = np.load(os.path.join(root_dir, 'spawrious224', 'metadata.npz'))
        y_array = metadata['y_array']
        bg_array = metadata['bg_array']

        encodings_data = np.load(
            os.path.join(self._root_dir, "spawrious", self._encoding_extractor, "combined.npz"))

        encodings = _put_encodings_in_right_order(
            indices_within_full_spawrious=np.arange(len(y_array)),
            encodings=encodings_data['encodings'],
            index_map=encodings_data['indices_map'],
        )

        self._full_spawrious = CustomExtractedSpawriousSubset(
            y_array=y_array,
            bg_array=bg_array,
            encodings=encodings,
        )

    def _split_train_val(
        self,
        train_ratio: float
    ) -> (CustomExtractedSpawriousSubset, CustomExtractedSpawriousSubset):

        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(len(self._full_spawrious))
        train_count = int(train_ratio * len(self._full_spawrious))
        train_indices = perm[:train_count]
        val_indices = perm[train_count:]

        train_ds = CustomExtractedSpawriousSubset(
            y_array=self._full_spawrious.y_array[train_indices],
            bg_array=self._full_spawrious.bg_array[train_indices],
            encodings=self._full_spawrious.encodings[train_indices],
        )

        val_ds = CustomExtractedSpawriousSubset(
            y_array=self._full_spawrious.y_array[val_indices],
            bg_array=self._full_spawrious.bg_array[val_indices],
            encodings=self._full_spawrious.encodings[val_indices],
        )

        return train_ds, val_ds

    def _select(
        self,
        ds: CustomExtractedSpawriousSubset,
        selected_background_ids: list[int],
        selected_class_ids: list[int],
    ) -> CustomExtractedSpawriousSubset:
        # select backgrounds
        bg_mask = np.zeros(len(ds), dtype=np.bool_)
        for b_id in selected_background_ids:
            bg_mask |= (ds.bg_array == b_id)

        # select classes
        class_mask = np.zeros(len(ds), dtype=np.bool_)
        for y_id in selected_class_ids:
            class_mask |= (ds.y_array == y_id)

        mask = (bg_mask & class_mask)

        return CustomExtractedSpawriousSubset(
            y_array=ds.y_array[mask],
            bg_array=ds.bg_array[mask],
            encodings=ds.encodings[mask],
        )

    def get_subset(self, split, *args, **kwargs) -> SpawriousSubsetExtracted:
        # train and val: mountain vs snow, corgi vs dachshund, then splitted by 80% vs 20% ratio.
        # test_diff_bg: beach vs desert, corgi vs dachshund
        # test_diff_class: mountain vs snow, bulldog vs labrador
        # test_all_diff: beach vs desert, bulldog vs labrador
        assert split in ['train', 'val', 'test_diff_bg', 'test_diff_class', 'test_all_diff']

        if split == 'train':
            ds, _ = self._split_train_val(train_ratio=0.8)
            ds = self._select(
                ds, selected_background_ids=[4, 5], selected_class_ids=[1, 2])
        elif split == 'val':
            _, ds = self._split_train_val(train_ratio=0.8)
            ds = self._select(
                ds, selected_background_ids=[4, 5], selected_class_ids=[1, 2])
        elif split == 'test_diff_bg':
            ds = self._select(
                self._full_spawrious, selected_background_ids=[0, 1], selected_class_ids=[1, 2])
        elif split == 'test_diff_class':
            ds = self._select(
                self._full_spawrious, selected_background_ids=[4, 5], selected_class_ids=[0, 3])
        elif split == 'test_all_diff':
            ds = self._select(
                self._full_spawrious, selected_background_ids=[0, 1], selected_class_ids=[0, 3])
        else:
            raise ValueError(f'Unexpected value {split=}')

        return SpawriousSubsetExtracted(
                ds=ds,
                reverse_task=self._reverse_task,
                sp_vector_to_add=self._sp_vector_to_add,
            )
