import os
import torch
import numpy as np
from tqdm import tqdm

from hydra.utils import instantiate
from omegaconf import DictConfig


def extract_encodings(config: DictConfig):
    """
    Extracts and saves encodings for a dataset.

    This function instantiates a data module and an encoding extractor as specified in the config.
    It then processes each batch of images from the data loader, extracting their encodings using the
    encoding extractor, and saves these encodings to disk.

    Args:
        config (DictConfig): Configuration object containing settings for datamodule,
                             encoding_extractor, save_path, and device.
    """
    # Instantiate the data module and encoding extractor from the configuration
    datamodule = instantiate(config.datamodule)
    datamodule.prepare_data()

    encoding_extractor = instantiate(config.encoding_extractor)

    # Set the encoding extractor to evaluation mode and move it to the specified device (e.g., GPU)
    encoding_extractor.eval()
    encoding_extractor.to(config.device)

    # Check if the data module has subsets and process accordingly
    if datamodule.with_subsets:
        dataloader, subset_names = datamodule.get_combined_dataloader()
        # Create a directory for each subset to store encodings
        for subset_name in subset_names:
            os.makedirs(os.path.join(config.save_path, subset_name), exist_ok=True)

        for (images, _, _, indices), _, dataloader_idx in tqdm(iter(dataloader), desc="Processing Images"):
            subset_name = subset_names[dataloader_idx]

            # Transfer images to the specified device
            images = images.to(config.device)

            # Extract and save encodings for the current batch
            _extract_and_save_encodings(images, indices, encoding_extractor,
                                        save_path=os.path.join(config.save_path, subset_name))
    else:
        dataloader = datamodule.get_dataloader()
        # Ensure the save directory exists
        os.makedirs(config.save_path, exist_ok=True)

        for images, indices in tqdm(dataloader, desc="Processing Images"):
            images = images.to(config.device)
            _extract_and_save_encodings(images, indices, encoding_extractor, save_path=config.save_path)


def _extract_and_save_encodings(images, indices, encoding_extractor, save_path):
    """
    Helper function to extract encodings for a batch of images and save them to disk.

    Args:
        images (torch.Tensor): Batch of images to process.
        indices (list): Indices of the images in the batch, used for naming the saved files.
        encoding_extractor (torch.nn.Module): Model used to extract encodings from the images.
        save_path (str): Directory where the extracted encodings will be saved.
    """
    # Extract encodings in a context that does not require gradient computation
    with torch.no_grad():
        encodings = encoding_extractor(images)

    # Convert encodings to numpy arrays and save each one individually
    encodings = encodings.detach().cpu().numpy()
    for encoding, idx in zip(encodings, indices):
        np.save(os.path.join(save_path, f"{idx}.npy"), encoding)
