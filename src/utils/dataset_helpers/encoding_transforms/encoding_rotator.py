import logging
import random
import numpy as np

from tqdm import tqdm
from scipy.stats import special_ortho_group

log = logging.getLogger(__name__)


class EncodingRotator:
    """
        A class that applies rotation to encoding vectors based on a class label.

        This class uses a set of predefined rotation matrices to transform encoding vectors.
        These rotation matrices are generated and stored statically to optimize performance,
        avoiding the computational cost of generating them for each instance initialization.
    """

    rotation_matrices = []

    @staticmethod
    def generate_rotation_matrices(n_matrices, emb_size):
        """
        Generates and stores a specified number of rotation matrices of a given size.

        This static method populates the `rotation_matrices` static list with randomly generated
        orthogonal matrices, which are used to rotate encoding vectors. It utilizes the
        special orthogonal group SO(n) to ensure the matrices are proper rotation matrices.

        Args:
            n_matrices (int): The number of rotation matrices to generate.
            emb_size (int): The dimensionality of the rotation matrices and the encoding vectors they will transform.
        """
        log.info(f"Generating {n_matrices} rotation matrices")
        EncodingRotator.rotation_matrices = [special_ortho_group.rvs(emb_size) for _ in tqdm(range(n_matrices))]

    def __init__(self):
        """
        Initializes an instance of EncodingRotator, sampling two rotation matrices from the static list.

        This constructor samples two distinct rotation matrices from the pre-generated static list
        of rotation matrices. These matrices are then used to rotate encoding vectors based on their
        class label.
        """
        self._rotation_matrices = random.sample(EncodingRotator.rotation_matrices, 2)

    def __call__(self, image_enc, class_label):
        """
                Applies a rotation transformation to an encoding vector based on the given class label.

                This method rotates an encoding vector using one of the two sampled rotation matrices
                associated with the instance. The choice of matrix is determined by the class label.

                Args:
                    image_enc (numpy.ndarray): The encoding vector to be rotated. This vector should match the dimensionality
                                               of the rotation matrices.
                    class_label (int): The class label determining which rotation matrix to use. Expected to be 0 or 1,
                                       corresponding to the two sampled matrices.

                Returns:
                    numpy.ndarray: The rotated encoding vector, having the same type and shape as the input vector.
                """
        return np.dot(image_enc, self._rotation_matrices[class_label].T).astype(image_enc.dtype)
