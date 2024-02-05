import logging
import random
import numpy as np

from tqdm import tqdm
from scipy.stats import special_ortho_group

log = logging.getLogger(__name__)


class EncodingRotator:
    """
    A class that applies random rotation to encoding vectors or matricesl.

    This class uses a set of predefined rotation matrices to transform encoding vectors or matrices.
    These rotation matrices are generated at initialization and stored to optimize performance,
    avoiding the computational cost of generating them for each rotation operation.
    """

    def __init__(self, n_matrices, emb_size):
        """
        Initializes an instance, creating a specified number of rotation matrices.

        This constructor generates a list of rotation matrices using the special orthogonal
        group SO(n), ensuring they are proper rotation matrices.

        Args:
            n_matrices (int): The number of rotation matrices to generate.
            emb_size (int): The dimensionality of the rotation matrices and the encoding vectors they will transform.
        """
        log.info(f"Generating {n_matrices} rotation matrices")
        self.rotation_matrices = [special_ortho_group.rvs(emb_size) for _ in tqdm(range(n_matrices))]

    def __call__(self, image_enc):
        """
        Applies a random rotation transformation to an encoding vector or matrix.

        This method rotates an encoding vector or matrix using a randomly selected rotation matrix
        from the instance's pre-generated list.

        Args:
            image_enc (numpy.ndarray): The encoding vector or matrix to be rotated. This vector should match the
                                       dimensionality of the rotation matrices.
        Returns:
            numpy.ndarray: The rotated encoding vector or matrix, having the same type and shape as the input.
        """
        rotation_matrix = random.choice(self.rotation_matrices)
        return np.dot(image_enc, rotation_matrix).astype(image_enc.dtype)
