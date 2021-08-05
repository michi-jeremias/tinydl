import torch
from torch.nn.modules.activation import ReLU
import torchvision
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
num_epochs = 20
learning_rate = 1e-2
batch_size = 256
momentum = 0.9
dropout = 0.5
weight_decay = 1e-4


architectures = {
    'A': [64, 'm', 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'],
    'B': [64, 64, 'm', 128, 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'],
    'D': [64, 64, 'm', 128, 128, 'm', 256, 256, 256, 'm', 512, 512, 512, 'm', 512, 512, 512, 'm'],
    'E': [64, 64, 'm', 128, 128, 'm', 256, 256, 256, 256, 'm', 512, 512, 512, 512, 'm', 512, 512, 512, 512, 'm'],
}


# Model definition
class VGG(nn.Module):
    """Creates a VGG architecture from a list.
    Each layer is followed by a ReLU. Kernel size and stride are fixed
    for all layers, see get_conv().

    Keyword arguments:
    vgg_architecture -- A list of numbers and letters, where a number
        denotes the output channels of a conv layer, and a letter 'm'
        adds a maxpool layer.
        [64, 128, 'm'] creates a network with two conv layers followed
        by a maxpool layer.
    num_channels -- Number of input channels (3 for RGB, 1 for
        greyscale).
    num_classes -- Number of output classes for classification.
    """

    def __init__(self, vgg_architecure, num_channels, num_classes):
        super(VGG, self).__init__()
        self.vgg_architecutre = vgg_architecure
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.conv = self.get_conv()
        self.fully = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fully(x)
        return x

    def get_conv(self):
        in_channels = self.num_channels
        conv = nn.Sequential()
        for block_idx, val in enumerate(self.vgg_architecutre):
            if type(val) == int:
                name = f'{block_idx}-conv{val}'
                block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=val,
                                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU())
                conv.add_module(module=block, name=name)
                in_channels = val
            elif val == 'm':
                name = f'{block_idx}-maxpool'
                block = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      nn.ReLU())
                conv.add_module(module=block, name=name)
        return conv


# Initialize model
model = VGG(vgg_architecure=architectures['A'], num_channels=3, num_classes=10)
x_trial = torch.randn(1, 3, 224, 224)
print(model(x_trial).shape)  # [1, 10]

# Loss, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(),
                      lr=learning_rate, weight_decay=weight_decay)
