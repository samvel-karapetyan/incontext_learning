import torch.nn as nn

from torchvision import models


class ResNet50(nn.Module):
    def __init__(self,
                 pretrained: bool,
                 *args, **kwargs):
        super().__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
