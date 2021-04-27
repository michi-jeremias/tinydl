"""Illustrate the use of tensorboard with a simple CNN on MNIST"""
# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from timefunc import timefunc

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
DATAPATH = '../data'
num_epochs = 5
batch_size = 1024
learning_rate = 0.001
num_classes = 10

# Transforms, dataset, dataloader
dataset = datasets.MNIST(root=DATAPATH, download=True)
train_mean = torch.mean(dataset.data * 1.) / 255
train_std = torch.std(dataset.data * 1.) / 255
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((train_mean), (train_std))])

train_dataset = datasets.MNIST(
    root=DATAPATH, train=True, transform=transform, download=True)
torch.manual_seed(0)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(
    root=DATAPATH, train=False, transform=transform, download=True)
torch.manual_seed(0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

single_loader = DataLoader(train_dataset, batch_size=1)
x, y = next(iter(single_loader))


# Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(
                3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(
                3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 16, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


model = CNN(num_classes=10).to(device=device)


# Loss, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Tensorboard writer
# CTRL-SHIFT-ö to open terminal
# tensorboard --logdir ../tensorboard/logs
TBPATH = '../tensorboard/logs'
writer = SummaryWriter(log_dir=f'{TBPATH}/MNIST')


# Train
@timefunc
def train():
    step = 0
    print('Start training')
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            # Forward
            score = model(data).to(device=device)
            loss = criterion(score, target).to(device=device)
            _, label = score.max(1)
            batch_accuracy = (target == label).sum() / data.shape[0]

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Step
            optimizer.step()

            writer.add_scalar(tag='Training loss',
                              scalar_value=loss, global_step=step)
            writer.add_scalar(tag='Batch accuracy',
                              scalar_value=batch_accuracy, global_step=step)
            step += 1
        print(f'Loss: {loss}]')
    print('Finished training')


train()


# Evaluate
@timefunc
def check_accuracy(loader, model):
    model.eval()
    if loader.dataset.train:
        print('Calculating train accuracy')
    else:
        print('Calculating test accuracy')

    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)
            values, labels = scores.max(1)
            num_correct += (targets == labels).sum()
            num_samples += values.shape[0]

    accuracy = num_correct / num_samples
    print(f'[{num_correct}/{num_samples}] - {accuracy}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
