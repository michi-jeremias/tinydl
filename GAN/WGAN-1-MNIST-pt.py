"""17-05-2021: WGAN (Wasserstein GAN)"""

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


# Hyperparameters, constants
DATAPATH = '../data'
TBLOGPATH = '../logs/tensorboard/WGAN-1/MNIST'
NUM_EPOCHS = 5
BATCH_SIZE = 64
Z_DIM = 100
IMG_CHANNELS = 1
IMG_SIZE = 64
LEARNING_RATE = 5e-5
WEIGHT_CLIP = 0.01
CRITIC_ITERATIONS = 5


# Models
def init_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(layer.weight, 0.0, 0.02)


class Critic(nn.Module):
    """Doesn't output sigmoid"""

    def __init__(self, img_channels):
        super().__init__()
        self.img_channels = img_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channels, out_channels=128,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=128),
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
                      kernel_size=4, stride=1, padding=0)
        )
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):

    def __init__(self, z_dim, img_channels):
        super().__init__()
        self.z_dim = Z_DIM
        self.img_channels = img_channels
        self.net = nn.Sequential(
            # n_out = (n_in - 1)*stride - 2*pad + kernel
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


C = Critic(img_channels=IMG_CHANNELS).to(device)  # Replaces Discriminator
G = Generator(z_dim=Z_DIM, img_channels=IMG_CHANNELS).to(device)


# Data
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)],
                         [0.5 for _ in range(IMG_CHANNELS)])
])
dataset = datasets.MNIST(root=DATAPATH, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Optimizer
opt_C = optim.RMSprop(params=C.parameters(), lr=LEARNING_RATE)
opt_G = optim.RMSprop(params=G.parameters(), lr=LEARNING_RATE)


# Loss
criterion = nn.BCELoss()


# Tensorboard
writer_fake = SummaryWriter(
    log_dir=TBLOGPATH + '/fake', comment='Fake data')
writer_real = SummaryWriter(
    log_dir=TBLOGPATH + '/real', comment='Real data')


@timefunc
def train():
    print('Start training.')
    tb_step = 0
    C.train()
    G.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (data, _) in enumerate(loader):
            current_bs = data.shape[0]
            real_img = data.to(device=device)

            # Critic will be trained more
            for _ in range(CRITIC_ITERATIONS):
                z = torch.rand(current_bs, Z_DIM, 1, 1).to(device=device)
                fake_img = G(z)
                scores_real = C(real_img)
                scores_fake = C(fake_img)
                loss_C = -(torch.mean(scores_real) - torch.mean(scores_fake))
                opt_C.zero_grad()
                loss_C.backward(retain_graph=True)
                opt_C.step()

                for p in C.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            # Generator
            # scores_fake_update = C(G(z)) # if we don't use retain_graph
            scores_fake_update = C(fake_img)
            loss_G = -torch.mean(scores_fake_update)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if batch_idx == 0:
                with torch.no_grad():
                    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]\t'
                          f'Loss C: {loss_C}\tLoss G: {loss_G}')
                    fake_imgs = torchvision.utils.make_grid(
                        tensor=fake_img[:32], padding=2, normalize=True)
                    real_imgs = torchvision.utils.make_grid(
                        tensor=real_img[:32], padding=2, normalize=True)
                    writer_fake.add_image(
                        tag='Fake', img_tensor=fake_imgs, global_step=tb_step)
                    writer_real.add_image(
                        tag='Real', img_tensor=real_imgs, global_step=tb_step)
        tb_step = 0
    print('Finished training.')


train()
