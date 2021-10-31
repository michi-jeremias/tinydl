"""
Code template by Aladdin Persson
Youtube: Pytorch Bidirectional LSTM example
https://www.youtube.com/watch?v=jGst43P-TJA
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

from timefunc import timefunc

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
DATAPATH = '../data'
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

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
        self.fc = nn.Linear(in_features=hidden_size
                            * 2, out_features=num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(device)     # hidden state
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(device)     # cell state
        out, _ = self.lstm(x, (h0, c0))  # _ is (hidden_state, cell_state)
        out = self.fc(out[:, -1, :])
        return out


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
            data = data.to(device=device).squeeze(1)  # Remove the channel dim
            targets = targets.to(device=device)

            # Forward
            scores = model(data).to(device)
            loss = criterion(scores, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Step
            optimizer.step()
    print('Training complete')


train()


# Evaluation
@timefunc
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on train set')
    else:
        print('Checking accuracy on test set')

    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device).squeeze(1)
            targets = targets.to(device)

            scores = model(data)
            values, labels = scores.max(1)
            num_correct += (labels == targets).sum()
            num_samples += data.shape[0]

    accuracy = num_correct / num_samples * 100
    print(f'Correct: {num_correct}/{num_samples}: {accuracy:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
