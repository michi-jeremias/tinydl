"""30-04-2021
Simple implementation of a GAN"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from timefunc import timefunc


# Model
class Discriminator(nn.Module):

    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features=img_dim, out_features=128),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # The discriminator returns just 'false' or 'true', hence the sigmoid
        return self.disc(x)


class Generator(nn.Module):

    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=256),
            nn.LeakyReLU(0.1),
            # For MNIST, the fake image will have dimension 28*28=784
            nn.Linear(in_features=256, out_features=img_dim),
            # The input images are normalized in the interval (-1, 1),
            # so the fake image will also be mapped to the same interval.
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATAPATH = '../data'
num_epochs = 20
learning_rate = 3e-4    # Best lr for use with Adam
batch_size = 32
z_dim = 64              # 32, 128, 256, ...
img_dim = 1 * 28 * 28


# Initialize models
torch.manual_seed(0)
disc = Discriminator(img_dim=img_dim).to(device)
gen = Generator(z_dim=z_dim, img_dim=img_dim).to(device)
fixed_noise = torch.randn(batch_size, z_dim).to(device)

mnist_mean = 0.1307
mnist_std = 0.3081
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(mnist_mean,), std=(mnist_std,))
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


# Data
dataset = datasets.MNIST(root=DATAPATH, transform=transform, download=True)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
optimizer_disc = optim.Adam(params=disc.parameters(), lr=learning_rate)
optimizer_gen = optim.Adam(params=gen.parameters(), lr=learning_rate)


# Loss
criterion = nn.BCELoss()


# Tensorboard
TBLOGPATH = '../logs/tensorboard/simple-gan/MNIST'
writer_fake = SummaryWriter(log_dir=f'{TBLOGPATH}/fake')
writer_real = SummaryWriter(log_dir=f'{TBLOGPATH}/real')


@timefunc
def train():
    tb_step = 0

    for epoch in range(num_epochs):
        for batch_idx, (real_img, _) in enumerate(loader):
            real_img = real_img.view(-1, 28 * 28).to(device=device)
            batch_size = real_img.shape[0]

            # Train discriminator: max log(D(real)) + log(1-D(G(z)))
            #                            disc_real       disc_fake
            # Discriminator returns a value between (0., 1.)
            noise = torch.randn(batch_size, z_dim).to(device=device)
            fake_img = gen(noise)
            disc_real = disc(real_img).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake_img).view(-1)
            # If we don't use retain_graph=True in loss_disc.backward:
            # disc_fake = disc(fake_img.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            # If we detach fake_img when calculating disc_fake
            # loss_disc.backward()
            loss_disc.backward(retain_graph=True)
            optimizer_disc.step()

            # Train gen: minimize log(1 - D(G(z))) <-> maximizelog(D(G(z)))
            # Minimizing will lead to gradient saturation so we will
            # maximize the equivalent expression
            output = disc(fake_img).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            # Tensorboard
            if batch_idx == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}] '
                    f'Loss D: {loss_disc:.4f} Loss G: {loss_gen:.4f}')

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real_img.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(
                        fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(
                        data, normalize=True)

                    writer_fake.add_image(
                        tag='MNIST Fake Images', img_tensor=img_grid_fake,
                        global_step=tb_step)
                    writer_real.add_image(
                        tag='MNIST Real Images', img_tensor=img_grid_real,
                        global_step=tb_step)
                    tb_step += 1


train()
