# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Create CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        # Same convolution: ((28-3+2*1) / 1 ) + 1 = 28
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        # Halves the dim size
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))

        # 2x MaxPool divides the input size by 4: (28/2)/2
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Sanity check model output
x = torch.randn(size=[64, 1, 28, 28])
model = CNN()
model(x).shape

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
DATAPATH = '../data'
transform = transforms.ToTensor()

# Load data
train_dataset = datasets.MNIST(
    root=DATAPATH, train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root=DATAPATH, train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = CNN().to(device=device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)


# Train network
def train():
    for epoch in range(num_epochs):
        print(f'Training epoch {epoch + 1}/{num_epochs}')
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data = data.to(device=device)
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


# Check accuracy
def check_accuracy(loader, model):
    model.eval()
    num_correct = 0
    num_samples = 0

    if loader.dataset.train:
        print('Calculating train accuracy')
    else:
        print('Calculating test accuracy')

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)
            maxscore, maxindex = scores.max(1)
            num_correct += ((maxindex == targets).sum())
            num_samples += data.shape[0]

    accuracy = float(num_correct) / float(num_samples)
    print(
        f'Correct: {num_correct}/{num_samples} /w accuracy: {100 * accuracy:.2f}%')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
