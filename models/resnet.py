import torch
import torch.nn as nn
from torchvision.models import resnet18
from functools import partial


def resnet_factory(resnet: torch.nn.Module,
                   n_channel: int = 3,
                   n_classes: int = 1000):
    """
    Factory for resnet models where the user can define the number of channels and classes

    :param resnet: A implementation of ResNet
    :type resnet: torch.nn.Module
    :param n_channel: Number of channels in the inputs, defaults to 3 (RGB)
    :type n_channel: int, optional
    :param n_classes: Number of output classes, defaults to 1000 (ImageNet)
    :type n_classes: int, optional
    """
    model = resnet(pretrained=False)
    model.conv1 = nn.Conv2d(n_channel,
                            64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
    model.fc = nn.Linear(512, n_classes)
    return model


def micro_resnet_factory(resnet: torch.nn.Module,
                         n_channel: int = 3,
                         n_classes: int = 1000):

    model = resnet(pretrained=False)
    model.conv1 = nn.Conv2d(1,
                            64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
    model.fc = nn.Linear(512, 8)
    return model
