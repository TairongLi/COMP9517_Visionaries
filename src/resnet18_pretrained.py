"""ResNet18 model setup for ImageNet transfer learning."""

import torch.nn as nn
from torchvision import models


def build_resnet18_pretrained(num_classes, freeze_backbone=False):
    """Create an ImageNet-pretrained ResNet18 with a project-specific classifier."""

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        # Freezing the backbone trains only the final classifier layer.
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
