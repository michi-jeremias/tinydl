"""03-05-2021 Simple GAN #2"""

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
num_hidden = 256
z_dim = 100


# Models
class Discriminator(nn.Module):

    def __init__(self, img_dim, num_hidden):
        super().__init__()
        self.img_dim = img_dim
        self.num_hidden = num_hidden
        self.net = nn.Sequential(
            nn.Linear(in_features=self.img_dim, out_features=self.num_hidden),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(num_features=self.num_hidden),
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
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.num_hidden),
            nn.Linear(self.num_hidden, self.img_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


modelG = Generator(img_dim=28 * 28, num_hidden=num_hidden, z_dim=z_dim)
modelG = modelG.to(device)
modelD = Discriminator(img_dim=28 * 28, num_hidden=num_hidden)
modelD = modelD.to(device)


# Data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root=DATAPATH, train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)


# Loss
criterion = nn.BCELoss()
optimizerG = optim.Adam(params=modelG.parameters(), lr=learning_rate)
optimizerD = optim.Adam(params=modelD.parameters(), lr=learning_rate)


# Tensorboard
tb_step = 0
writer_fake = SummaryWriter(log_dir=f'{TBLOGPATH}/fake')
writer_real = SummaryWriter(log_dir=f'{TBLOGPATH}/real')


# Train
def train():
    tb_step = 0
    for epoch in range(num_epochs):
        modelD.train()
        modelG.train()
        print('Start training.')

        for batch_idx, (real_data, _) in enumerate(train_loader):
            current_batch_size = real_data.shape[0]
            latent_noise = torch.randn(current_batch_size, z_dim).to(device)
            real_data = real_data.reshape(current_batch_size, -1).to(device)

            fake_data = modelG(latent_noise).to(device)
            # Forward and loss for the discriminator
            D_fake = modelD(fake_data)
            D_real = modelD(real_data)
            ones = torch.ones_like(D_real).to(device)
            zeros = torch.zeros_like(D_fake).to(device)
            D_loss_fake = criterion(D_fake, zeros)
            D_loss_real = criterion(D_real, ones)
            D_loss = D_loss_fake + D_loss_real

            # Backward and step discriminator
            optimizerD.zero_grad()
            D_loss.backward(retain_graph=True)
            optimizerD.step()

            # Forward and loss for the generator
            G_fake = modelG(latent_noise)
            G_loss = criterion(G_fake, zeros)

            # Backwand and step generator
            optimizerG.zero_grad()
            G_loss.backward()
            optimizerG.step()

            # Tensorboard
            if batch_idx == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}]'
                    f'D Loss: {D_loss}\tG Loss: {G_loss}')

                with torch.no_grad():
                    fake_img = modelG(latent_noise).reshape(-1, 1, 28, 28)
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


train()
