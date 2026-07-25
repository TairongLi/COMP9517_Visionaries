"""ResNet18 model setup for training from random initialisation."""

import torch.nn as nn
from torchvision import models


def build_resnet18_scratch(num_classes):
    """Create a ResNet18 without pretrained weights."""

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
