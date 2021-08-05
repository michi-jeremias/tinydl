"""Pretrained Alexnet on MNIST data."""

# Imports
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms.transforms import Grayscale
import torch.optim as optim

from timefunc import timefunc

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
torch.manual_seed(0)
DATAPATH = '../data'
num_epochs = 5
learning_rate = 0.001
num_classes = 10
batch_size = 1024

# Data
dataset = MNIST(root=DATAPATH, train=True,
                transform=transforms.ToTensor(), download=True)
train_mean = torch.mean(dataset.data * 1.) / 255
train_std = torch.std(dataset.data * 1.) / 255

# Convert to 3 channel grayscale so we don't need to add a Conv2d(1, 3)
# in the beginning ofthe model.
# Upsample data to shape [batch_size, 3, 64, 64].

# MNIST mean/std normalization
transform = transforms.Compose([
    transforms.Resize(
        (64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)])

# Imagenet mean/std normalization
transform = transforms.Compose([
    transforms.Resize(
        (64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# No normalization +++
transform = transforms.Compose([
    transforms.Resize(
        (64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()])

train_dataset = MNIST(root=DATAPATH, train=True,
                      transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_dataset = MNIST(root=DATAPATH, train=False,
                     transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)
single_loader = DataLoader(dataset=train_dataset, batch_size=1)

# Model load and adaptation
model = models.alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
model = model.to(device=device)

# Loss, optimizer
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
        print(f'Epoch [{epoch+1}/{num_epochs}] - loss: {loss}')
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
            _, label = score.max(1)
            num_correct += (label == target).sum()
            num_samples += data.shape[0]
    accuracy = num_correct / num_samples
    print(f'[{num_correct}/{num_samples}] - {100 * accuracy:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
