"""06-05-2021 Simple GAN #3.
Added batchnorm layers to discriminator and generator, and initialized
models with Xavier init. Doesn't really work out.
"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import timefunc
from utils import del_logs

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
DATAPATH = '../data'
num_epochs = 20
num_hidden = 128
img_dim = 28 * 28
z_dim = 64
batch_size = 64
learning_rate = 3e-4


# Tensorboard
TBLOGPATH = '../logs/tensorboard/simple-gan-3/MNIST'
writer_real = SummaryWriter(log_dir=f'{TBLOGPATH}/real')
writer_fake = SummaryWriter(log_dir=f'{TBLOGPATH}/fake')


# Xavier
def xavier_init(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)


# Models
class Discriminator(nn.Module):

    def __init__(self, img_dim, num_hidden):
        super().__init__()
        self.img_dim = img_dim
        self.num_hidden = num_hidden
        self.net = nn.Sequential(
            nn.Linear(in_features=self.img_dim,
                      out_features=self.num_hidden),
            nn.BatchNorm1d(num_features=self.num_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=self.num_hidden, out_features=1),
            nn.Sigmoid()
        )
        self.net.apply(xavier_init)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):

    def __init__(self, img_dim, num_hidden, z_dim):
        super().__init__()
        self.img_dim = img_dim
        self.num_hidden = num_hidden
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(in_features=self.z_dim,
                      out_features=self.num_hidden),
            nn.BatchNorm1d(num_features=self.num_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=self.num_hidden, out_features=self.img_dim),
            nn.Sigmoid()
        )
        self.net.apply(xavier_init)

    def forward(self, x):
        return self.net(x)


D = Discriminator(img_dim=img_dim, num_hidden=num_hidden).to(device)
G = Generator(img_dim=img_dim, num_hidden=num_hidden, z_dim=z_dim).to(device)


# Data
transform = transforms.ToTensor()
dataset = datasets.MNIST(root=DATAPATH, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Loss, opzimizer
criterion = nn.BCELoss()
optim_D = optim.Adam(params=D.parameters(), lr=learning_rate)
optim_G = optim.Adam(params=G.parameters(), lr=learning_rate)


# Train
@timefunc
def train():
    try:
        print('Start training.')
        tb_step = 0
        G.train()
        D.train()

        for epoch in range(num_epochs):
            for batch_idx, (x, _) in enumerate(loader):
                current_batch_size = x.shape[0]
                x = x.reshape(current_batch_size, -1).to(device)
                z = torch.randn(current_batch_size, z_dim).to(device)
                fake_image = G(z)
                score_real = D(x)
                score_fake = D(G(z))

                # Loss Discriminator
                optim_D.zero_grad()
                loss_D_real = criterion(
                    score_real, torch.ones_like(score_real))
                loss_D_fake = criterion(
                    score_fake, torch.zeros_like(score_fake))
                loss_D = loss_D_fake + loss_D_real
                loss_D.backward(retain_graph=True)
                optim_D.step()

                # Loss Generator
                optim_G.zero_grad()
                fake_score_update = D(fake_image)
                loss_G = criterion(fake_score_update,
                                   torch.ones_like(fake_score_update))
                loss_G.backward()
                optim_G.step()

                if batch_idx == 0:
                    print(
                        f'Epoch [{epoch+1}/{num_epochs}]\t'
                        f'D Loss: {loss_D:.6f}\tG Loss: {loss_G:.6f}')

                    with torch.no_grad():
                        fake_img = G(z).reshape(-1, 1, 28, 28)
                        real_img = x.reshape(-1, 1, 28, 28)
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
    except KeyboardInterrupt:
        print('Manually interrupted.')


del_logs('../logs/tensorboard/simple-gan-3')


train()
