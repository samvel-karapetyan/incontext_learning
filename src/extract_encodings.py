import os
import torch
import numpy as np
from tqdm import tqdm

from hydra.utils import instantiate
from omegaconf import DictConfig

torch.multiprocessing.set_sharing_strategy('file_system')  # avoiding the limits on open file descriptors


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

    if datamodule.with_subsets:
        dataloader, subset_names = datamodule.get_combined_dataloader()

        # Prepare separate encoding extractors and savers for each subset.
        encoding_extractor_and_savers = {}
        for subset_name in subset_names:
            save_path = os.path.join(config.save_path, subset_name)
            encoding_extractor_and_saver = EncodingsExtractorAndSaver(save_path=save_path,
                                                                      encoding_extractor=encoding_extractor)
            encoding_extractor_and_savers[subset_name] = encoding_extractor_and_saver

        # Process each batch from the dataloader, extract encodings.
        for (images, indices), _, dataloader_idx in tqdm(iter(dataloader), desc="Processing"):
            subset_name = subset_names[dataloader_idx]
            encoding_extractor_and_savers[subset_name].extract_encodings(images.to(config.device) if type(images) == torch.Tensor else images, indices)

        # Save all extracted encodings after processing.
        for encoding_extractor_and_saver in encoding_extractor_and_savers.values():
            encoding_extractor_and_saver.save_all()

    else:
        dataloader = datamodule.get_dataloader()

        encoding_extractor_and_saver = EncodingsExtractorAndSaver(save_path=config.save_path,
                                                                  encoding_extractor=encoding_extractor)

        for images, indices in tqdm(dataloader, desc="Processing"):
            encoding_extractor_and_saver.extract_encodings(images.to(config.device) if type(images) == torch.Tensor else images, indices)

        encoding_extractor_and_saver.save_all()


class EncodingsExtractorAndSaver:
    """
        A utility class to extract and save encodings from images.

        This class handles the extraction of encodings using a provided encoding extractor model,
        aggregates these encodings, and saves them to disk.

        Attributes:
            save_path (str): The path where the encodings are saved.
            encoding_extractor (Model): The model used to extract encodings from images.
        """

    def __init__(self, save_path, encoding_extractor):
        """
        Initializes the EncodingsExtractorAndSaver.

        Args:
            save_path (str): The directory path where the encodings and index mappings will be saved.
            encoding_extractor (Model): The model to use for encoding extraction.
        """
        self._all_indices = []  # Aggregator for indices of processed images.
        self._all_encodings = []  # Aggregator for encodings extracted from images.
        self._save_path = save_path  # Path to save encodings.
        self._encoding_extractor = encoding_extractor  # Model for encoding extraction.

        os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists.

    def extract_encodings(self, images, indices):
        """
        Extracts encodings for a batch of images and stores them.

        Args:
            images (Tensor): A batch of images to process.
            indices (Tensor): Indices of the images in the batch.
        """
        with torch.no_grad():
            encodings = self._encoding_extractor(images)

        # Convert encodings to numpy arrays and save each one individually
        encodings = encodings.detach().cpu().numpy()

        self._all_indices.append(indices)
        self._all_encodings.append(encodings)

    def save_all(self):
        """
        Saves all accumulated encodings and their corresponding indices to disk.

        This method consolidates all extracted encodings and their indices collected during the encoding extraction process.
        It saves a combined .npz file containing the encodings and a mapping of the original image indices to their positions
        in the accumulated array. This mapping facilitates quick lookup of encodings based on the original dataset indices.

        The saved .npz file contains two arrays:
        - 'encodings': An array of all extracted encodings.
        - 'indices_map': An array where each position corresponds to an index in the original dataset, and its value is the
          position of that index in the 'encodings' array.

        Note: This method should be called after all batches of images have been processed through `extract_encodings`.
        """
        # Concatenate all indices and encodings collected during processing
        indices = np.concatenate(self._all_indices)
        encodings = np.concatenate(self._all_encodings)

        # Generating a mapping array for quick index lookup
        indices_map = np.full(indices.max() + 1, -1)  # Initialize with -1 to indicate unmapped indices
        indices_map[indices] = np.arange(len(indices))

        # Save encodings and the index mapping to a .npz file for efficient storage and access
        np.savez(os.path.join(self._save_path, "combined.npz"), encodings=encodings, indices_map=indices_map)
