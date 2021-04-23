"""
VGG16 Implementation - ConvNet D
Following Aladdin Persson
https://www.youtube.com/watch?v=ACmuBbuXn20
"""

# Imports
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from dlio import load_checkpoint, save_checkpoint
from timefunc import timefunc

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
DATAPATH = '../data'
input_size = 224
num_classes = 10
learning_rate = 0.001
batch_size = 64


# Model definition
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv64_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))
        self.conv64_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))

        self.conv128_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))
        self.conv128_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))

        self.conv256_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))
        self.conv256_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))
        self.conv256_3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))

        self.conv512_1 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))
        self.conv512_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))
        self.conv512_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))

        self.conv512_4 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))
        self.conv512_5 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))
        self.conv512_6 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1))

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.lin1 = nn.Linear(in_features=7 * 7 * 512, out_features=4096)
        self.lin2 = nn.Linear(in_features=4096, out_features=4096)
        self.lin3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x = self.relu(self.conv64_1(x))
        x = self.relu(self.conv64_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv128_1(x))
        x = self.relu(self.conv128_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv256_1(x))
        x = self.relu(self.conv256_2(x))
        x = self.relu(self.conv256_3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv512_1(x))
        x = self.relu(self.conv512_2(x))
        x = self.relu(self.conv512_3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv512_4(x))
        x = self.relu(self.conv512_5(x))
        x = self.relu(self.conv512_6(x))
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(self.relu(self.lin1(x)))
        x = self.dropout(self.relu(self.lin2(x)))
        x = self.relu(self.lin3(x))
        return x


# Initialize model
model = VGG16()
trial_x = torch.randn(1, 3, 224, 224)

# Data
train_dataset = datasets.MNIST(root=DATAPATH, train=True,
                               transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root=DATAPATH, train=False,
                              transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Train

# Evaluate

# Save model
