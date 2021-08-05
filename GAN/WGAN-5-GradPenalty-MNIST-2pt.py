"""29-06-2021: WGAN with Gradient Penalty
See https://arxiv.org/abs/1704.00028"""

# Import
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import del_logs, timefunc


# Hyperparameters
DATAPATH = '../data'
TBLOGPATH = '../logs/tensorboard/WGAN/WGAN-5-GP-MNIST'
NUM_EPOCHS = 5
BATCHSIZE = 64
IMG_SIZE = 64
IMG_CHANNELS = 3
ADAM_BETAS = (0.0, 0.9)
LEARNING_RATE = 4e-4
ZDIM = 100


# Models
def init_weights(layer):
    if isinstance(
        layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d,
                nn.InstanceNorm2d)):
        nn.init.normal_(layer, 0.0, 0.02)


class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channels, out_channels=128,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(num_features=256),
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(num_features=512),
            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(num_features=1024),
            nn.Conv2d(in_channels=1024, out_channels=1,
                      kernel_size=4, stride=1, padding=0)
        )
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):

    def __init__(self, z_dim, img_channels):
        super().__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.z_dim, out_channels=1024,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.ConvTranspose2d(in_channels=512, out_channels=256,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.ConvTranspose2d(in_channels=128, out_channels=self.img_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.img_channels),
            nn.Tanh()
        )
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)
