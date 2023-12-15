import os
import logging
import numpy as np

from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.utils.dataset_helpers import TokenGenerator

log = logging.getLogger(__name__)


def generate_and_save_val_sets(config: DictConfig):
    """
    Generates and saves validation sets based on the given configuration.

    This function instantiates a data module, sets it up, and then iterates
    over the validation data loader to save each batch along with the associated
    instance data.

    Args:
        config (DictConfig): A Hydra configuration object containing settings for data generation.
    """
    # Instantiate and set up the data module
    datamodule = instantiate(config.datamodule)
    datamodule.setup()

    # Load tokens data
    tokens_data = load_tokens_data(config)

    # Create tokens generators
    tokens_generator = TokenGenerator(tokens_data=tokens_data,
                                      are_spurious_tokens_fixed=config.are_spurious_tokens_fixed,
                                      are_class_tokens_fixed=config.are_class_tokens_fixed)

    spurious_tokens_generator, class_tokens_generator = tokens_generator()

    # Prepare save paths for each validation set
    save_paths = prepare_save_paths(datamodule, config)

    # Process and save each batch in the validation dataloader
    val_dataloader = datamodule.val_dataloader()
    for batch, batch_idx, dataloader_idx in tqdm(iter(val_dataloader), total=len(val_dataloader)):
        save_path = save_paths[dataloader_idx]
        batch_size = len(batch[0])  # batch[0] contains the first element of each tuple in the batch
        for i in range(batch_size):
            global_i = batch_idx * batch_size + i
            input_seq, spurious_labels, class_labels, image_indices = [batch[x][i] for x in range(4)]
            spurious_tokens = next(spurious_tokens_generator)
            class_tokens = next(class_tokens_generator)

            # Stack relevant data for saving
            instance = np.stack([image_indices, spurious_labels, class_labels], axis=1)

            # Save data as a compressed numpy file
            np.savez(os.path.join(save_path, f"{global_i}.npz"),
                     spurious_tokens=spurious_tokens,
                     class_tokens=class_tokens,
                     instance=instance)


def load_tokens_data(config: DictConfig) -> dict:
    """
    Loads tokens data from a specified path.

    Args:
        config (DictConfig): A Hydra configuration object.

    Returns:
        dict: A dictionary containing tokens data.
    """
    tokens_data_path = os.path.join(config.dataset_path, "avg_norms", f"{config.encoding_extractor}_l2.npz")
    tokens_data = np.load(tokens_data_path, mmap_mode="r")

    return {k: tokens_data[k] for k in tokens_data.keys()}


def prepare_save_paths(datamodule, config: DictConfig) -> dict:
    """
    Prepares and returns save paths for validation sets.

    Args:
        datamodule: The instantiated data module.
        config (DictConfig): A Hydra configuration object.

    Returns:
        dict: A dictionary mapping dataloader indices to save paths.
    """
    save_paths = {}
    for idx, val_set_name in enumerate(datamodule.ValSets):
        set_save_path = os.path.join(config.save_path, val_set_name.value)
        os.makedirs(set_save_path, exist_ok=True)  # Changed exist_ok to True to avoid error if the directory exists
        save_paths[idx] = set_save_path

    return save_paths
