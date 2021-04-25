"""Pretrained VGG16 (D) on MNIST
Checking train accuracy
[56833/60000] - 94.7217%
Function: check_accuracy, time: 9.20s
Checking test accuracy
[9434/10000] - 94.3400%
"""

# Imports
import torch
import torch.nn as nn
from torch.nn.modules.upsampling import Upsample
import torchvision.datasets as datasets
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import Normalize
import torchvision.models as models

from timefunc import timefunc

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
torch.manual_seed(0)
DATAPATH = '../data'
batch_size = 1024
num_epochs = 5
learning_rate = 0.001
num_classes = 10

# Data
transform = transforms.Compose(
    [transforms.Resize(size=(32, 32), interpolation=transforms.InterpolationMode.BILINEAR),
     transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
transform = transforms.Compose(
    [transforms.Resize(size=(32, 32), interpolation=transforms.InterpolationMode.NEAREST),
     transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
transform = transforms.Compose(
    [transforms.Resize(size=(32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
     transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(
    root=DATAPATH, train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root=DATAPATH, train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

single_loader = DataLoader(train_dataset, 1)

# Model
# - Load pretrained VGG16 and freeze the layers
# - Add a Conv2d(1, 3) to the front so the greyscale images work with
#   the VGG architecture.
# - Make the avgpool layer a pass through layer, because with input
#   size 28 and 5 max pools the output shape of features will be
#   [batch_size, 1, 1] and the pooling won't work.
# - Replace the first linear layer with a linear layer to match the
#   output size of features, and add nn.Linear(4096, 10) to get output
#   of shape [batch_size, 10] to match the 10 classes of MNIST.
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.features = nn.Sequential(nn.Conv2d(
    in_channels=1, out_channels=3, kernel_size=1, stride=1), *list(model.features))
model.avgpool = nn.Identity()
model.classifier[0] = nn.Linear(1 * 1 * 512, 4096)
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

# Optimizer, loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)


# Train
@timefunc
def train():
    model.train()
    print('Start training')
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            # Forward
            score = model(data)
            loss = criterion(score, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Step
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}] loss: {loss}')
    print('Finished training')


train()


# Evaluate
@timefunc
def check_accuracy(loader, model):
    model.eval()
    num_correct = 0
    num_samples = 0

    if loader.dataset.train:
        print('Checking train accuracy')
    else:
        print('Checking test accuracy')
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            score = model(data)
            value, label = score.max(1)
            num_correct += (label == target).sum()
            num_samples += data.shape[0]
    accuracy = num_correct / num_samples
    print(f'[{num_correct}/{num_samples}] - {100 * accuracy:.4f}%')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
