"""09-05-2021: DCGAN implementation,
see https://arxiv.org/abs/1511.06434 for the paper."""

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
batch_size = 128
num_epochs = 5
learning_rate = 2e-4
beta1 = 0.5
beta2 = 0.999
z_dim = 100
img_channels = 1
num_features = 8
IMG_SIZE = 64  # MNIST


# Models
def init_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d)):
        nn.init.normal_(tensor=layer.weight, mean=0.0, std=0.02)


class Discriminator(nn.Module):

    def __init__(self, img_channels):
        super().__init__()
        self.img_channels = img_channels
        self.net = nn.Sequential(
            # 64 to 32
            nn.Conv2d(in_channels=self.img_channels, out_channels=128,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            # 32 to 16
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=256),
            # 16 to 8
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=512),
            # 8 to 4
            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(1024),
            # 4 to 1
            nn.Conv2d(in_channels=1024, out_channels=1,
                      kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels):
        super().__init__()
        self.img_channels = img_channels
        self.z_dim = z_dim
        # n_out = (n_in - 1)*stride-2*pad_in + kernel_size + pad_out
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.z_dim, out_channels=1024,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=1024),
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
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)


D = Discriminator(img_channels=1).to(device=device)
G = Generator(z_dim=z_dim, img_channels=1).to(device=device)


# Test
def test():
    N, img_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    z = torch.rand(N, img_channels, H, W)
    D = Discriminator(img_channels)
    assert (D(z)).shape == (N, 1, 1, 1)
    print('nice')


# Data
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(img_channels)],
                         [0.5 for _ in range(img_channels)])
])
dataset = datasets.MNIST(root=DATAPATH, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Loss
criterion = nn.BCELoss()

# Optimizer
optim_D = optim.Adam(params=D.parameters(),
                     lr=learning_rate, betas=(beta1, beta2))
optim_G = optim.Adam(params=G.parameters(),
                     lr=learning_rate, betas=(beta1, beta2))


# Training
@timefunc
def train():
    print('Start training.')
    fake_writer = SummaryWriter(log_dir=f'{TBLOGPATH}/fake')
    real_writer = SummaryWriter(log_dir=f'{TBLOGPATH}/real')
    D.train()
    G.train()
    tb_step = 0

    for epoch in range(num_epochs):

        for batch_idx, (real_data, _) in enumerate(loader):
            current_batch_size = real_data.shape[0]
            x = real_data.to(device=device)
            # Random uniform
            z = torch.rand(current_batch_size, z_dim, 1, 1).to(device)
            fake_data = G(z)

            # Discriminator forward
            real_scores = D(x).reshape(-1)
            fake_scores = D(G(z)).reshape(-1)
            real_loss_D = criterion(real_scores, torch.ones_like(real_scores))
            fake_loss_D = criterion(fake_scores, torch.zeros_like(fake_scores))
            loss_D = (real_loss_D + fake_loss_D) / 2

            # Discriminator backward
            optim_D.zero_grad()
            loss_D.backward(retain_graph=True)  # retain_graph=True
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
                        f'\tLoss D: {loss_D:.6f}\tLoss G: {loss_G:.6f}')
                    fake_data = G(z).reshape(-1, 1, IMG_SIZE, IMG_SIZE)
                    fake_grid = torchvision.utils.make_grid(
                        fake_data[:32], normalize=True)
                    fake_writer.add_image(
                        'Fake Img', fake_grid, global_step=tb_step)
                    real_data = real_data.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
                    real_grid = torchvision.utils.make_grid(
                        real_data[:32], normalize=True)
                    real_writer.add_image(
                        'Real Img', real_grid, global_step=tb_step)
                    tb_step += 1
    print('Finished training.')


train()
