"""29-04-2021
Tensorboard showcase with hparam dictionaries.
"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from timefunc import timefunc


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
DATAPATH = 'data'
num_epochs = 5


# Data
dataset = datasets.MNIST(root=DATAPATH, download=True)
train_mean = torch.mean(dataset.data * 1.) / 255
train_std = torch.std(dataset.data * 1.) / 255
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])
train_dataset = datasets.MNIST(
    root=DATAPATH, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(
    root=DATAPATH, train=False, transform=transform, download=True)


# Model
class CNN(nn.Module):

    def __init__(self, num_classes=10):
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
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 16, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=self.num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


# Loss
criterion = nn.CrossEntropyLoss()


# Tensorboard
TBLOGPATH = 'logs/tensorboard'
learning_rates = [0.001]
batch_sizes = [1024]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [64, 512, 1024]
learning_rates = [0.01, 0.001]
batch_sizes = [64, 1024]


def train(model, optimizer, loader, writer, tb_step):
    model.train()

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

        # Statistics for tensorboard
        _, label = score.max(1)
        batch_accuracy = 100 * (label == target).sum() / data.shape[0]

        # Tensorboard
        writer.add_scalar(
            'Accuracy', scalar_value=batch_accuracy, global_step=tb_step)
        writer.add_scalar('Train loss', scalar_value=loss, global_step=tb_step)
        tb_step += 1

    return loss.item(), batch_accuracy


@timefunc
def tb_train():
    print('Starting hyperparameter search.')

    for lr in learning_rates:

        for bs in batch_sizes:
            print(f'LR {lr}, BS {bs}')
            writer = SummaryWriter(
                log_dir=f'{TBLOGPATH}/MNIST_bs-{bs}_lr-{lr}')

            # Reset model, optimizer and dataloader for the next setup
            # in the hyperparameter search
            torch.manual_seed(0)
            model = CNN().to(device)
            optimizer = optim.Adam(params=model.parameters(), lr=lr)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=bs, shuffle=True)
            tb_step = 0

            for epoch in range(num_epochs):
                accuracies = []
                losses = []
                loss, accuracy = train(model=model, optimizer=optimizer,
                                       loader=train_loader, writer=writer,
                                       tb_step=tb_step)

                # Generate tensorboard output
                writer.add_hparams(
                    hparam_dict={'learningrate': lr, 'batchsize': bs},
                    metric_dict={'accuracy': accuracy, 'loss': loss})
                print(f'Epoch [{epoch+1}/{num_epochs}] - loss: {loss:.4f}')

    print('Finished hyperparameter search.')


tb_train()
