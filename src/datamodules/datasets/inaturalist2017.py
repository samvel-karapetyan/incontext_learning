import logging
import os.path

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.utils.dataset_helpers import INaturalist2017Preparator

log = logging.getLogger(__name__)


class INaturalist2017Dataset(Dataset, INaturalist2017Preparator):
    """
    A custom dataset class for the iNaturalist 2017 dataset.

    This class extends PyTorch's Dataset and includes the INaturalist2017Preparator for handling
    the specific preprocessing requirements of the iNaturalist 2017 dataset. It supports loading
    images, applying necessary transformations, and providing the transformed images along with their IDs.

    Args:
        dataset_path (str): The file path to the dataset directory.
        min_images_per_category (int): The minimum number of images required per category.
        resize_size (int): The size to which the smaller edge of the image will be resized.
        crop_size (int): The size of the square crop to be extracted from the center of the image.
        fully_outer_supercategories (list of str): List of supercategories that should be entirely classified as 'outer'. All categories under these supercategories will be assigned to an 'outer' split.
        fully_inner_supercategories (list of str): List of supercategories that should be entirely classified as 'inner'. All categories under these supercategories will be assigned to an 'inner' split.
        outer_classes_size (float): A decimal representing the proportion of categories within a non-fully classified supercategory to be randomly assigned to the 'outer' split. The rest will be assigned to the 'inner' split.
    """

    def __init__(self,
                 dataset_path,
                 min_images_per_category,
                 resize_size,
                 crop_size,
                 fully_outer_supercategories,
                 fully_inner_supercategories,
                 outer_classes_size):
        # Initialize the INaturalist2017Preparator
        INaturalist2017Preparator.__init__(self, dataset_path,
                                           min_images_per_category,
                                           fully_outer_supercategories,
                                           fully_inner_supercategories,
                                           outer_classes_size)

        # Initialize the Dataset
        Dataset.__init__(self)

        # Define image transformations
        self._transform = transforms.Compose([
            transforms.Resize(resize_size),  # Resize the smaller edge to {resize_size} while maintaining aspect ratio
            transforms.CenterCrop(crop_size),  # Center crop to {crop_size}x{crop_size} pixels
            transforms.ToTensor(),  # Convert to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Normalize with ImageNet mean and std
        ])

        self._dataset_path = dataset_path

    def __getitem__(self, idx):
        """
        Retrieves an image and its ID at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its corresponding image ID.
        """
        # Access the data for the given index
        item = self._dataframe.iloc[idx]

        image_id = item["image_id"]
        image_path = os.path.join(self._dataset_path, item["image_path"])

        # Load the image from the file system
        image = Image.open(image_path).convert('RGB')

        # Apply the predefined transformations
        image = self._transform(image)

        # Return the transformed image and its ID
        return image, image_id

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self._dataframe)
