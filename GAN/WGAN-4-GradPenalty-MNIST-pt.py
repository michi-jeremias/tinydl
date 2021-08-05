"""04-06-2021: WGAN with Gradient Penalty - MNIST
See https://arxiv.org/abs/1704.00028"""

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


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters, etc.
DATAPATH = '../data'
TBLOGPATH = '../logs/tensorboard/WGAN/WGAN-4-GP-MNIST'
NUM_EPOCHS = 5
CRITIC_ITER = 5
IMG_SIZE = 64
IMG_CHANNELS = 1
ADAM_BETAS = (0.0, 0.9)
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
Z_DIM = 100
LAMBDA_GP = 10


# Models
def init_weights(layer):
    if isinstance(layer, (nn.BatchNorm2d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, 0.0, 0.02)


def gradient_penalty(critic, fake, real, device='cpu'):
    batch_size, c, w, h = real.shape
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    scores_interpolated = critic(interpolated)
    # grad_outputs is the size of the gradient output
    scores_gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=scores_interpolated,
        grad_outputs=torch.ones_like(scores_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    scores_gradient = scores_gradient.view(batch_size, -1)
    gradient_norm = scores_gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty


class Critic(nn.Module):
    """Doesn't output Sigmoid. Uses LayerNorm2d (InstanceNorm2d)
    instead of BatchNorm2d, so it doesn't normalize across batches"""

    def __init__(self, img_channels):
        super().__init__()
        self.img_channels = img_channels
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
    """Generates fake img from noise input"""

    def __init__(self, img_channels, z_dim):
        super().__init__()
        self.img_channels = img_channels
        self.z_dim = z_dim
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


C = Critic(IMG_CHANNELS).to(device=device)
G = Generator(IMG_CHANNELS, Z_DIM).to(device=device)


# Data
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)],
                         [0.5 for _ in range(IMG_CHANNELS)])
])
data = datasets.MNIST(root=DATAPATH, transform=transform, download=True)
loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)


# Optimizer
opt_C = optim.Adam(params=C.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
opt_G = optim.Adam(params=G.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)


# Tensorboard
writer_fake = SummaryWriter(log_dir=TBLOGPATH + '/fake', comment='Fake Writer')
writer_real = SummaryWriter(log_dir=TBLOGPATH + '/real', comment='Real Writer')


# Train
@timefunc
def train():
    print('Start training.')
    C.train()
    G.train()
    tb_step = 0

    for epoch in range(NUM_EPOCHS):

        for batch_idx, (data, _) in enumerate(loader):
            current_bs = data.shape[0]
            real_img = data.to(device=device)

            for iteration in range(CRITIC_ITER):
                z = torch.randn(current_bs, Z_DIM, 1, 1).to(device=device)
                fake_img = G(z)
                scores_real = C(real_img)
                scores_fake = C(fake_img)
                scores_interpolated = gradient_penalty(
                    C, fake_img, real_img, device)
                loss_C = -(torch.mean(scores_fake) - torch.mean(scores_real)
                           ) + LAMBDA_GP * scores_interpolated
                opt_C.zero_grad()
                loss_C.backward(retain_graph=True)
                opt_C.step()

            scores_fake_update = C(fake_img)
            loss_G = -torch.mean(scores_fake_update)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]'
                      f'\tLoss C: {loss_C:.4f}\tLoss G: {loss_G:.4f}')
                with torch.no_grad():
                    real_grid = torchvision.utils.make_grid(
                        tensor=real_img[:32], padding=2, normalize=True)
                    fake_grid = torchvision.utils.make_grid(
                        tensor=fake_img[:32], padding=2, normalize=True)
                    writer_real.add_image(
                        tag='Real img',
                        img_tensor=real_grid,
                        global_step=tb_step)
                    writer_fake.add_image(
                        tag='Fake img',
                        img_tensor=fake_grid,
                        global_step=tb_step)
                tb_step += 1

    print('Finish training.')


train()
