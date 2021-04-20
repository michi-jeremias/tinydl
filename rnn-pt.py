# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.optim as optim


# Hyperparameters
# Images with 28x28 pixels, number of featuers foreach time step is 28
input_size = 28
sequence_length = 28    # Tx - sequence length of each input sequence
num_layers = 2          # RNN layers
hidden_size = 256       # Number of nodes in each time step
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model with sanity check
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True: the inputs will have the batch size as
        # first dimension. The dimension will then be m*time_seq*features
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size
                            * sequence_length, out_features=num_classes)

    def forward(self, x):
        # First activation layer
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        # _ is the hidden state but we don't store it here
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


x = torch.randn(batch_size, 28, 28)
model = RNN(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, num_classes=num_layers)
x.shape

model(x).shape


# Load data
DATAPATH = '../data/'
train_set = datasets.MNIST(root=DATAPATH, train=True,
                           transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = datasets.MNIST(root=DATAPATH, train=False, transform=transforms.ToTensor(),
                          download=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Initialize model
model = RNN(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, num_classes=num_classes).to(device=device)

# Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)


# Train
def train():
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for batch_index, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            # RNN needs data to be of shape [batchsize, 28, 28]
            data = data.squeeze(1)
            targets = targets.to(device=device)

            # Forward
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Step
            optimizer.step()
    print('Finished training')


train()


# Evaluate
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    if loader.dataset.train:
        print('Checking train accuracy')
    else:
        print('Checking test accuracy')

    model.eval()
    for data, targets in loader:
        data = data.to(device)
        data = data.squeeze(1)
        targets = targets.to(device)

        scores = model(data)
        maxvalue, maxindex = scores.max(1)
        num_correct += int((maxindex == targets).sum())
        num_samples += scores.shape[0]

    accuracy = num_correct / num_samples

    model.train()

    print(f'Accuracy [{num_correct}/{num_samples}]: {accuracy * 100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
