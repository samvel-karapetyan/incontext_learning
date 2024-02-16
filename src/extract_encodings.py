import os
import torch
import numpy as np
from tqdm import tqdm

from hydra.utils import instantiate
from omegaconf import DictConfig

torch.multiprocessing.set_sharing_strategy('file_system') # avoiding the limits on open file descriptors

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
    encoding_extractor = instantiate(config.encoding_extractor)

    # Retrieve the data loader from the data module
    dataloader = datamodule.get_dataloader()

    # Create the directory specified in the config to save the encodings
    os.makedirs(config.save_path, exist_ok=True)

    # Set the encoding extractor to evaluation mode and move it to the specified device (e.g., GPU)
    encoding_extractor.eval()
    encoding_extractor.to(config.device)

    all_indices = []
    all_encodings = []
    # Iterate over batches of images and their corresponding indices with a progress bar
    for images, indices in tqdm(dataloader, desc="Processing Images"):
        # Transfer images to the specified device
        images = images.to(config.device)

        # Perform encoding extraction without computing gradients
        with torch.no_grad():
            encodings = encoding_extractor(images)

        # Detach encodings from the current graph and move them to CPU
        encodings = encodings.detach().cpu().numpy()

        all_encodings.append(encodings)
        all_indices.append(indices)

    all_indices = np.concatenate(all_indices)
    all_encodings = np.concatenate(all_encodings)

    # Create a mapping from original image indices to their positions in the all_indices array.
    # This facilitates quick lookup of encodings based on the original dataset indices.
    indices_map = np.full(all_indices.max() + 1, 0)
    indices_map[all_indices] = np.arange(len(all_indices))

    np.savez(os.path.join(config.save_path, f"combined.npz"), encodings=all_encodings, indices_map=indices_map)