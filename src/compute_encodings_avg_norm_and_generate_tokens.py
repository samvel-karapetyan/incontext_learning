import os
import logging
import numpy as np

from tqdm import tqdm
from omegaconf import DictConfig
from typing import List

log = logging.getLogger(__name__)


def compute_encodings_avg_norm_and_generate_tokens(config: DictConfig) -> None:
    """
    Compute the average norm of encoding vectors and generate tokens.

    This function calculates the L2 norm of each encoding vector stored as a numpy array in files within
    a specified directory, averages these norms, and saves the average norm. Additionally, it generates
    random fixed tokens based on the average norm.

    Parameters:
    - config (DictConfig): Configuration object containing `data_path` and `encoding_extractor` keys.
    """
    # Construct the path to the directory with encoding files
    encodings_path = os.path.join(config.data_path, config.encoding_extractor)
    if config.split:
        encodings_path = os.path.join(encodings_path, config.split)

    norms: List[float] = []  # List to store norms of each encoding

    # Process each file in the encodings directory
    enc = None
    for filename in tqdm(os.listdir(encodings_path)):
        # Load the encoding from the file
        enc = np.load(os.path.join(encodings_path, filename))
        # Compute and append the L2 norm of the encoding
        norms.append(np.linalg.norm(enc))

    # Calculate the average of all norms
    avg_norm = np.mean(norms)
    log.info(f"Average norm: {avg_norm}")

    # Generate random fixed tokens
    fixed_tokens = generate_fixed_tokens(avg_norm.item(), len(enc))

    # Save the average norm and tokens
    save_dir = os.path.join(config.data_path, "avg_norms")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config.encoding_extractor}_l2.npz")

    np.savez(save_path, avg_norm=np.array(avg_norm), token_len=np.array(len(enc)), **fixed_tokens)


def generate_fixed_tokens(norm: float, token_len: int) -> dict:
    """
    Generate fixed tokens based on a given norm and token length.

    Parameters:
    - norm (float): Norm to scale the random tokens.
    - token_len (int): Length of each token.

    Returns:
    - dict: Dictionary containing two types of random and two types of opposite fixed tokens.
    """
    # Generating random tokens
    tokens = np.random.randn(4, token_len).astype(np.float32)
    # Normalize and scale each token array
    tokens = (tokens / np.linalg.norm(tokens, axis=1, keepdims=True)) * norm
    # Split into two types of tokens
    random_tokens = {
        "random_class_tokens": tokens[:2],
        "random_spurious_tokens": tokens[2:]
    }

    # Generating opposite tokens
    first_rows = np.random.randn(2, token_len).astype(np.float32)
    tokens = np.vstack((first_rows, -first_rows))
    # Normalize and scale each token array
    tokens = (tokens / np.linalg.norm(tokens, axis=1, keepdims=True)) * norm
    # Split into two types of tokens
    opposite_tokens = {
        "opposite_class_tokens": tokens[::2],
        "opposite_spurious_tokens": tokens[1::2]
    }

    return dict(**random_tokens, **opposite_tokens)
