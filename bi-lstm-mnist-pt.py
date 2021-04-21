# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

from timefunc import timefunc

# Hyperparameters
DATAPATH = '../data'
learning_rate = 0.001
batch_size = 64
num_classes = 10
input_size = 28
hidden_size = 256
num_epochs = 2
sequence_length = 28
num_layers = 2

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
train_dataset = datasets.MNIST(
    root=DATAPATH, train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root=DATAPATH, train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)


# Network: create a bidirectional RNN
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        pass


# Initialize network
model = BRNN(input_size=input_size, hidden_size=hidden_size,
             num_layers=num_layers, num_classes=num_classes)

# Criterion, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train
@timefunc
def train():
    model.train()
    for epoch in range(num_epochs):
        print(f'Training epoch [{epoch + 1}/{num_epochs}]')
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward
            scores = model(data).to(device)
            loss = criterion(scores, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Step
            optimizer.step()


# Evaluation
