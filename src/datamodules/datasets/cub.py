import os
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CUBDataset(Dataset):
    """
    A PyTorch Dataset class for the Caltech-UCSD Birds (CUB) 200-2011 dataset. This class provides mechanisms to load,
    transform, and access the bird images and their corresponding IDs from the dataset.

    Args:
        dataset_path (str): The path to the dataset directory containing 'images.txt' and the 'images' folder.
        resize_size (int): The size to which the smaller edge of the image will be resized.
        crop_size (int): The size of the square crop to be extracted from the center of the image.
    """
    def __init__(self, dataset_path, resize_size, crop_size):
        super(CUBDataset, self).__init__()

        self._dataset_path = dataset_path
        self._df = pd.read_csv(os.path.join(dataset_path, 'images.txt'), sep=' ', header=None, names=['id', 'filepath'])

        # Define image transformations
        self._transform = transforms.Compose([
            transforms.Resize(resize_size),  # Resize the smaller edge to {resize_size} while maintaining aspect ratio
            transforms.CenterCrop(crop_size),  # Center crop to {crop_size}x{crop_size} pixels
            transforms.ToTensor(),  # Convert to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Normalize with ImageNet mean and std
        ])

    def __getitem__(self, idx):
        """
        Retrieves an image and its ID by index.

        Args:
            idx (int): The index of the image in the dataset.

        Returns:
            tuple: A tuple containing the transformed image as a PyTorch Tensor and its corresponding image ID.
        """
        # Access the data for the given index
        item = self._df.iloc[idx]

        image_id = item["id"]
        image_path = os.path.join(self._dataset_path, 'images', item["filepath"])

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
        return len(self._df)
