import torch
import torch.nn as nn


class DinoV2(nn.Module):
    """
    DinoV2 is a PyTorch Module for extracting embeddings using the DINO V2 architecture.

    This class wraps around a DINO V2 model loaded from Torch Hub, enabling it to be used
    as a standard PyTorch Module. The forward method passes input through the DINO V2 model
    to obtain embeddings.

    Args:
        model_name (str): A string identifier for the DINO V2 model variant to be loaded.
    """

    def __init__(self, model_name, *args, **kwargs):
        super().__init__()

        # Load the specified DINO V2 model from Torch Hub
        self.model = torch.hub.load("facebookresearch/dinov2", model=model_name)

    def forward(self, x):
        """
        Forward pass for embedding extraction.

        Args:
            x (torch.Tensor): The input tensor to the model. Typically, this should be an image
                              or a batch of images.

        Returns:
            torch.Tensor: The embeddings produced by the DINO V2 model for the input.
        """
        # Pass the input through the DINO V2 model to obtain embeddings
        emb = self.model(x)

        return emb
