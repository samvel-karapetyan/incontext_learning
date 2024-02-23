import logging
import os.path

import numpy as np

from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from wilds.datasets.wilds_dataset import WILDSSubset

log = logging.getLogger(__name__)


class CustomizedWaterbirdsDataset(WaterbirdsDataset):
    """
    A customized version of the WaterbirdsDataset from the WILDS dataset collection.
    This class extends the standard WaterbirdsDataset with some dataset-specific
    functionalities and modifications.
    """

    def __init__(self, root_dir, data_type="img", encoding_extractor=None, *args, **kwargs):
        """
        Initializes the CustomizedWaterbirdsDataset instance.

        Args:
            root_dir (str): The root directory of the dataset.
        """
        super().__init__(root_dir=root_dir, *args, **kwargs)
        self._root_dir = root_dir
        self._data_type = data_type
        self._encoding_extractor = encoding_extractor

        if data_type == "encoding":
            # Prepare encodings and data files
            self._encodings_data = {}
            for split in self.DEFAULT_SPLITS:
                self._encodings_data[split] = {}

                encodings_data = np.load(os.path.join(root_dir, "waterbirds", encoding_extractor, split, "combined.npz"))
                self._encodings_data[split]["encodings"] = encodings_data["encodings"]
                self._encodings_data[split]["indices_map"] = encodings_data["indices_map"]

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image data (x), label (y), spurious label (c), and index (idx).
        """
        x, y, metadata = super().__getitem__(idx)
        x, y, c = x, y, metadata[0]

        return x, y, c, idx

    def get_input(self, idx):
        if self._data_type == "img":
            return super().get_input(idx)
        elif self._data_type == "encoding":
            split_dict = {value: key for key, value in self.split_dict.items()}
            set_name = split_dict[self.split_array[idx]]

            # Retrieve the set information from the encodings data
            set_info = self._encodings_data[set_name]

            # Get the index map for the current set
            index_map = set_info["indices_map"]

            # Find the actual index in the encodings using the index map
            actual_index = index_map[idx]

            # Finally, access the encoding using the actual index
            x = set_info["encodings"][actual_index]
            return x

    def get_subset(self, split, frac=1.0, transform=None):
        """
        Generates a  subset of the dataset based on the specified split.

        Note: This method and the CustomizedWildsSubset class will be removed when PR https://github.com/p-lambda/wilds/pull/156 is merged.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")

        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        if frac < 1.0:
            # Randomly sample a fraction of the split
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        return CustomizedWildsSubset(self, split_idx, transform)


class CustomizedWildsSubset(WILDSSubset):
    """
    A customized subset of the WILDS dataset, used to handle specific data retrieval and transformations.
    This class extends the standard WILDSSubset with some dataset-specific functionalities.

    Note: This class will be removed when PR https://github.com/p-lambda/wilds/pull/156 is merged.
    """

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index from the subset.

        Args:
            idx (int): The index of the item to retrieve from the subset.

        Returns:
            tuple: A tuple containing the transformed image data (x), label (y), and other metadata.
        """
        vv = self.dataset[self.indices[idx]]

        x, y, *_ = vv
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        return x, y, *vv[2:]
