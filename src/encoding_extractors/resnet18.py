from torchvision import models
import torch
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x
