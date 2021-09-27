import torch.nn as nn


def init_xavier(layer):
    if isinstance(
            layer,
            (nn.Linear, nn.BatchNorm2d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)


def init_normal(layer):
    if isinstance(
            layer,
            (nn.Linear, nn.BatchNorm2d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, 0.0, 0.02)
