import numpy as np


class IdentityTransform:
    """
    Returns the input numpy array without modification. Useful as a non-altering step in data processing pipelines.
    """

    def __call__(self, image_enc: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Returns the input numpy array unchanged.

        Args:
            image_enc (np.ndarray): Input numpy array.

        Returns:
            np.ndarray: Unchanged input array.
        """
        return image_enc
