"""12-05-2021: DCGAN (Deep Convolution GAN) Implementation for the\
CIPHAR10 dataset"""

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

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
DATAPATH = '../data/'
TBLOGPATH = '../logs/tensorboard/DCGAN/CIPHAR10'
NUM_EPOCHS = 30
IMG_CHANNELS = 3
IMG_SIZE = 64
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
ADAM_BETAS = (0.5, 0.999)
Z_DIM = 100


# Models
def init_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(tensor=layer.weight, mean=0.0, std=0.02)


class Discriminator(nn.Module):

    def __init__(self, img_channels):
        super().__init__()
        self.img_channels = img_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channels, out_channels=128,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=1024),
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
        self.z_dim = z_dim
        self.img_channels = img_channels
        # ConvTranspose2d out = (n_in - 1) * stride - 2*pad + kernel_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.z_dim, out_channels=1024,
                kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=1024),
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512,
                kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256,
                kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128,
                kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=self.img_channels,
                kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)


D = Discriminator(img_channels=IMG_CHANNELS).to(device=device)
G = Generator(z_dim=Z_DIM, img_channels=IMG_CHANNELS).to(device=device)


# Data
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)],
                         [0.5 for _ in range(IMG_CHANNELS)])])
dataset = datasets.CIFAR10(root=DATAPATH, transform=transform, download=True)
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


# Loss
criterion = nn.BCELoss()


# Optimizer
opt_D = optim.Adam(params=D.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
opt_G = optim.Adam(params=G.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)


# Tensorboard
writer_fake = SummaryWriter(log_dir=f'{TBLOGPATH}/fake')
writer_real = SummaryWriter(log_dir=f'{TBLOGPATH}/real')


@timefunc
def train():
    print('Start training.')
    tb_step = 0
    D.train()
    G.train()

    for epoch in range(NUM_EPOCHS):

        for batch_idx, (data, _) in enumerate(loader):
            current_bs = data.shape[0]
            real_img = data.to(device=device)
            z = torch.rand(current_bs, Z_DIM, 1, 1).to(device=device)
            fake_img = G(z)

            # Discriminator
            scores_real = D(real_img)
            loss_D_real = criterion(scores_real, torch.ones_like(scores_real))
            scores_fake = D(fake_img)
            loss_D_fake = criterion(scores_fake, torch.zeros_like(scores_real))
            loss_D = loss_D_fake + loss_D_real
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Generator
            # fake_scores_updated = D(fake_img)
            fake_scores_updated = D(G(z))
            loss_G = criterion(fake_scores_updated,
                               torch.ones_like(fake_scores_updated))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if batch_idx == 0:
                with torch.no_grad():
                    fake_grid = torchvision.utils.make_grid(
                        tensor=fake_img[:32], padding=2, normalize=True)
                    writer_fake.add_image(
                        tag='Fake Images',
                        img_tensor=fake_grid,
                        global_step=tb_step)

                    real_grid = torchvision.utils.make_grid(
                        tensor=real_img[:32], padding=2, normalize=True)
                    writer_real.add_image(
                        tag='Real Images',
                        img_tensor=real_grid,
                        global_step=tb_step)

                    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]\t'
                          f'Loss D: {loss_D:.6f}\tLoss G: {loss_G:.6f}')

        tb_step += 1

    print('Finished training.')


train()


del_logs(TBLOGPATH)
