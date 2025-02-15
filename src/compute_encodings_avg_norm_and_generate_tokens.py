import os
import logging
import numpy as np

from omegaconf import DictConfig

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

    # Load the encoding from the file
    enc = np.load(os.path.join(encodings_path, "train", "combined.npz"))['encodings']

    # Calculate the average of all norms
    avg_norm = np.linalg.norm(enc, axis=1).mean()
    log.info(f"Average norm: {avg_norm}")

    # Generate random fixed tokens
    token_len = enc.shape[1]
    fixed_tokens = generate_fixed_tokens(
        norm=avg_norm.item(),
        token_len=token_len,
    )

    # Save the average norm and tokens
    save_dir = os.path.join(config.data_path, "avg_norms")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config.encoding_extractor}_l2.npz")

    np.savez(save_path,
             avg_norm=np.array(avg_norm),
             token_len=np.array(token_len),
             **fixed_tokens)


def generate_fixed_tokens(norm: float, token_len: int) -> dict:
    """
    Generate fixed tokens based on a given norm and token length.

    Parameters:
    - norm (float): Norm to scale the random tokens.
    - token_len (int): Dimensionality of each token.

    Returns:
    - dict: Dictionary containing three pairs of random and opposite fixed tokens.
    """

    # Generating random tokens
    tokens = np.random.randn(6, token_len).astype(np.float32)
    tokens = (tokens / np.linalg.norm(tokens, axis=1, keepdims=True)) * norm
    # Split into 3 types of tokens
    random_tokens = {
        "random_class_tokens": tokens[:2],
    }

    # Generating opposite tokens
    first_rows = np.random.randn(3, token_len).astype(np.float32)
    tokens = np.vstack((first_rows, -first_rows))
    tokens = (tokens / np.linalg.norm(tokens, axis=1, keepdims=True)) * norm
    # Split into 3 types of tokens
    opposite_tokens = {
        "opposite_class_tokens": tokens[::3],
    }

    return dict(**random_tokens, **opposite_tokens)
