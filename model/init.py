import torch
import torch.nn as nn


def init_xavier(layer, fixed_seed=False):
    if isinstance(
            layer,
            (nn.Linear, nn.BatchNorm2d, nn.Conv2d, nn.ConvTranspose2d)):
        if fixed_seed:
            torch.manual_seed(0)
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)


def init_normal(layer, fixed_seed=False):
    if isinstance(
            layer,
            (nn.Linear, nn.BatchNorm2d, nn.Conv2d, nn.ConvTranspose2d)):
        if fixed_seed:
            torch.manual_seed(0)
        nn.init.normal_(layer.weight, 0.0, 0.02)
        layer.bias.data.fill_(0)
