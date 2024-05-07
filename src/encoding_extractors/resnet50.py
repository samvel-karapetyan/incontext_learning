import torch.nn as nn

from torchvision import models


class ResNet50(nn.Module):
    """
    ResNet50 is a PyTorch Module for feature extraction using the ResNet-50 architecture.

    This class wraps a pre-trained ResNet-50 model from torchvision.models, loading weights
    trained on the ImageNet dataset. It removes the final linear layer to enable feature
    extraction from input images.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the ResNet50 module.

        Loads the ResNet-50 model with pre-trained weights from the ImageNet dataset
        and creates a new module without the final linear layer for feature extraction.
        """
        super().__init__()

        # Load the ResNet-50 model with pre-trained weights
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Create a module without the final linear layer
        self.without_linear = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        """
        Performs a forward pass through the ResNet50 module.

        Args:
            x (torch.Tensor): The input tensor to the module, typically an image or a batch of images.

        Returns:
            torch.Tensor: The extracted features from the input images.
        """
        # Pass the input through the ResNet-50 model without the final linear layer
        emb = self.without_linear(x).squeeze((2, 3))

        return emb
