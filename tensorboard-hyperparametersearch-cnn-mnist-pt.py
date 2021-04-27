"""26-04-2021
Example of a simple (and not so correct) hyperparameter search
with tensorboard. Custom CNN on MNIST data.
"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from timefunc import timefunc

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
DATAPATH = '../data'
num_epochs = 5
num_classes = 10


# Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(8, 16, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


model = CNN(num_classes=10).to(device=device)


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

test_dataset = datasets.MNIST(
    root=DATAPATH, train=False, transform=transform, download=True)
torch.manual_seed(0)


# Loss
criterion = nn.CrossEntropyLoss()

# Training with tensorboard logging
batch_sizes = [64, 128, 1024]
batch_sizes = [64, 1024]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
learning_rates = [0.1, 0.001]
TB_LOGPATH = '../logs/tensorboard/MNIST'

batch_size = 1024
learning_rate = 0.001
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)


@timefunc
def train(model, optimizer, loader, writer):
    tb_step = 0
    model.train()
    print('Start training')

    for epoch in range(num_epochs):
        for data, target in loader:
            data = data.to(device=device)
            target = target.to(device=device)

            # Forward
            score = model(data).to(device=device)
            loss = criterion(score, target).to(device=device)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Step
            optimizer.step()

            # Tensorboard
            _, label = score.max(1)
            batch_accuracy = 100 * (label == target).sum() / data.shape[0]
            # writer.add_scalar('Training loss', scalar_value=loss)
            # writer.add_scalar('Batch accuracy', scalar_value=batch_accuracy)

            tb_step += 1
        print(f'Epoch [{epoch+1}/{num_epochs}] - loss: {loss:.2f}')


@timefunc
def log_train():
    for batch_size in batch_sizes:
        print(f'Batch size: {batch_size}')
        torch.manual_seed(0)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True)

        for learning_rate in learning_rates:
            model = CNN(num_classes=10).to(device=device)
            optimizer = optim.Adam(
                params=model.parameters(), lr=learning_rate)
            writer = SummaryWriter(
                log_dir=f'{TB_LOGPATH}_bs{batch_size}_lr{learning_rate}')
            print(f'Learning rate: {learning_rate}')

            train(model=model, optimizer=optimizer,
                  loader=train_loader, writer=writer)
    print('End log_train')


log_train()


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
