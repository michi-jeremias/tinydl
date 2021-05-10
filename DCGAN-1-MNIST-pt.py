"""09-05-2021: DCGAN implementation."""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import del_logs, timefunc


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
DATAPATH = '../data'
TBLOGPATH = '../logs/tensorboard/DCGAN/MNIST'
batch_size = 64
num_epochs = 20
learning_rate = 3e-4
z_dim = 64
img_channels = 3
num_features = 8


# Models
def init_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(layer.weight, 0.0, 0.02)


class Discriminator(nn.Module):

    def __init__(self, img_channels, num_features):
        super().__init__()
        self.img_channels = img_channels
        self.num_features = num_features
        # Input shape is [batch_size, 3, 64, 64]
        # Output shape is [batch_size, 1]
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=self.img_channels,
                out_channels=self.num_features,
                kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(num_features, num_features * 2, 4, 2, 1),
            self._block(num_features * 2, num_features * 4, 4, 2, 1),
            self._block(num_features * 4, num_features * 8, 4, 2, 1),
            self._block(num_features * 8, 1, 4, 2, 0),
            nn.Sigmoid()
        )
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )


class Generator(nn.Module):

    def __init__(self, z_dim, img_channels, num_features):
        super().__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.num_features = num_features
        # Input shape: [batch_size, z_dim]
        # Output: [batch_size, num_channels=3, 64, 64]
        self.net = nn.Sequential(
            self._block(z_dim, num_features * 16, 4, 1, 0),
            self._block(num_features * 16, num_features * 8, 4, 2, 1),
            self._block(num_features * 8, num_features * 4, 4, 2, 1),
            self._block(num_features * 4, num_features * 2, 4, 2, 1),
            nn.ConvTranspose2d(num_features * 2, self.img_channels, 4, 2, 1),
            nn.Tanh()
        )
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


D = Discriminator(img_channels, num_features).to(device=device)
G = Generator(z_dim, img_channels, num_features).to(device=device)

# Test


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    z = torch.randn(N, in_channels, H, W)
    D = Discriminator(in_channels, 8)
    assert (D(z)).shape == (N, 1, 1, 1)
    print('nice')


# Data
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.MNIST(root=DATAPATH, tansform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Loss
criterion = nn.BCELoss()

# Optimizer
optim_D = optim.Adam(params=D.parameters(), lr=learning_rate)
optim_G = optim.Adam(params=G.parameters(), lr=learning_rate)


# Training
def train():
    print('Start training.')
    fake_writer = SummaryWriter(log_dir=f'{TBLOGPATH}/fake')
    real_writer = SummaryWriter(log_dir=f'{TBLOGPATH}/real')
    D.train()
    G.train()
    tb_step = 0

    for epoch in range(num_epochs):

        for batch_idx, (real_data, _) in loader:
            current_batch_size = real_data.shape[0]
            x = real_data.reshape(current_batch_size, -1).to(device=device)
            z = torch.randn(current_batch_size, z_dim).to(device)
            fake_data = G(z)

            # Discriminator forward
            real_scores = D(x)
            fake_scores = D(G(z))
            real_loss_D = criterion(real_scores, torch.ones_like(real_scores))
            fake_loss_D = criterion(fake_scores, torch.zeros_like(fake_scores))
            loss_D = real_loss_D + fake_loss_D

            # Discriminator backward
            optim_D.zero_grad()
            loss_D.backward()  # retain_graph=True
            optim_D.step()

            # Generator forward
            fake_scores_updated = D(G(z))
            loss_G = criterion(fake_scores_updated,
                               torch.ones_like(fake_scores_updated))

            # Generator backward
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            if batch_idx == 0:
                with torch.no_grad():
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}]'
                        f'\tLoss D: {loss_D:.6f}\tLoss G:{loss_G:.6f}')
                    fake_data = G(z).reshape(-1, 1, 28, 28)
                    fake_grid = torchvision.utils.make_grid(
                        fake_data, normalize=True)
                    fake_writer.add_image(
                        'Fake Img', fake_grid, global_step=tb_step)
                    real_data = real_data.reshape(-1, 1, 28, 28)
                    real_grid = torchvision.utils.make_grid(
                        real_data, normalize=True)
                    real_writer.add_image(
                        'Real Img', real_grid, global_step=tb_step)
                    tb_step += 1
