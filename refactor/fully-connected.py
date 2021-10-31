# imports
import torch
import torch.nn as nn
# Functions without parameters (relu, tanh, ...)
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Create fully connected network


class NN(nn.Module):
    # input_size = 784 = 28*28 (MNIST)
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Sanity check if the model returns the correct shape on some random
# data
model = NN(784, 10)
x = torch.randn(64, 784)
print(model(x).shape)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load data
DATAPATH = '../data/'
train_dataset = datasets.MNIST(
    root=DATAPATH, train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root=DATAPATH, train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device=device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):
    for batch_index, (data, targets) in enumerate(train_loader):
        # Move data to cuda if available
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Reshape data of shape [batchsize, 1, 28, 28] to [batchsize, 784]
        data = data.reshape(data.shape[0], -1)

        # Forward
        scores = model(data)
        loss = criterion(input=scores, target=targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent/adam step
        optimizer.step()


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
    num_correct = 0.
    num_samples = 0.

    # Disables certain layers for calculation like batchnorm
    model.eval()

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device=device)
            y = y.to(device=device)
            X = X.reshape(X.shape[0], -1)

            scores = model(X)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += X.size(0)

        accurary = num_correct / num_samples
        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


check_accuracy(test_loader, model)
