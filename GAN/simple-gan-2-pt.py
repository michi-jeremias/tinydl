"""03-05-2021 Simple GAN #2.
Replaced the Tanh() with a Sigmoid() compared with simple-gan
and removed the normalization with mean 0.5 and std 0.5.
The normalization makes the data centered (roughly) around 0, therefore
the generator has to be able to output negative values, which the Tanh
does.
Without the normalization the Tanh is not necessary anymore.
"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
DATAPATH = '../data'
TBLOGPATH = '../logs/tensorboard/simple-gan-2/MNIST/simple-gan-2'
batch_size = 64
learning_rate = 3e-4
num_epochs = 20
num_hidden = 128
z_dim = 64


# Models (with BN)
class Discriminator(nn.Module):

    def __init__(self, img_dim, num_hidden):
        super().__init__()
        self.img_dim = img_dim
        self.num_hidden = num_hidden
        self.net = nn.Sequential(
            nn.Linear(in_features=self.img_dim, out_features=self.num_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.num_hidden, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):

    def __init__(self, img_dim, num_hidden, z_dim):
        super().__init__()
        self.img_dim = img_dim
        self.num_hidden = num_hidden
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, self.num_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(self.num_hidden, self.img_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


D = Discriminator(img_dim=28 * 28, num_hidden=128).to(device)
G = Generator(img_dim=28 * 28, num_hidden=256, z_dim=64).to(device)


# Data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root=DATAPATH, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)


# Loss
criterion = nn.BCELoss()
optimizerD = optim.Adam(params=D.parameters(), lr=learning_rate)
optimizerG = optim.Adam(params=G.parameters(), lr=learning_rate)


# Tensorboard
tb_step = 0
writer_fake = SummaryWriter(log_dir=f'{TBLOGPATH}/fake')
writer_real = SummaryWriter(log_dir=f'{TBLOGPATH}/real')


# Train
def train():
    tb_step = 0
    D.train()
    G.train()
    print('Start training.')

    for epoch in range(num_epochs):

        for batch_idx, (real_data, _) in enumerate(train_loader):
            current_batch_size = real_data.shape[0]
            latent_noise = torch.randn(current_batch_size, z_dim).to(device)
            real_data = real_data.reshape(current_batch_size, -1).to(device)
            fake_data = G(latent_noise).to(device)

            # Forward and loss for the discriminator
            scores_fake = D(fake_data)
            scores_real = D(real_data)
            ones = torch.ones_like(scores_real).to(device)
            zeros = torch.zeros_like(scores_fake).to(device)
            D_loss_fake = criterion(scores_fake, zeros)
            D_loss_real = criterion(scores_real, ones)
            D_loss = (D_loss_fake + D_loss_real)

            # Backward and step discriminator
            optimizerD.zero_grad()
            D_loss.backward(retain_graph=True)
            optimizerD.step()

            # Forward and loss for the generator
            scores_fake_updated = D(fake_data)
            G_loss = criterion(scores_fake_updated,
                               torch.ones_like(scores_fake_updated))

            # Backwand and step generator
            optimizerG.zero_grad()
            G_loss.backward()
            optimizerG.step()

            # Tensorboard
            if batch_idx == 0:

                print(
                    f'Epoch [{epoch+1}/{num_epochs}]\t'
                    f'D Loss: {D_loss:.6f}\tG Loss: {G_loss:.6f}')

                with torch.no_grad():
                    fake_img = G(latent_noise).reshape(-1, 1, 28, 28)
                    real_img = real_data.reshape(-1, 1, 28, 28)
                    grid_fake = torchvision.utils.make_grid(
                        fake_img, normalize=True)
                    grid_real = torchvision.utils.make_grid(
                        real_img, normalize=True)
                    writer_fake.add_image(
                        tag='Fake Images', img_tensor=grid_fake,
                        global_step=tb_step)
                    writer_fake.add_image(
                        tag='Real Images', img_tensor=grid_real,
                        global_step=tb_step)
                    tb_step += 1

    print('Finished training.')


train()
