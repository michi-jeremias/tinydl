# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from timefunc import timefunc

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
torch.manual_seed(0)
DATAPATH = '../data'
batch_size = 1024
learning_rate = 1e-3
num_classes = 10
num_epochs = 5

# Load pretrained model, train only changed layers
# Model adaptation to fit CIFAR10 (3x32x32 images, 10 output classes)
# After the last MaxPool2d in model.features the output size will be
# [batch_size, 1, 1, 512].
# Therefore avgpool is replaced with nn.Identity(), and an additional
# layer with a mapping from [batch_size, 1*1*512] to [batch_size, 25088]
# is added in order to work with the classifier part of the model.
model = models.vgg16(pretrained=True)
model.avgpool = nn.Identity()
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Linear(512, 4096),
    *list(model.classifier.children())[1:6],
    nn.Linear(4096, num_classes))
model = model.to(device=device)

# Model without pretraining, train whole model
model = models.vgg16(pretrained=False)
model.avgpool = nn.Identity()
model.classifier = nn.Sequential(
    nn.Linear(512, 4096),
    *list(model.classifier.children())[1:6],
    nn.Linear(4096, num_classes))
model = model.to(device=device)

# Data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose(
    [transforms.ToTensor()])


train_dataset = datasets.CIFAR10(
    root=DATAPATH, train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(
    root=DATAPATH, train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

# Shape check
single_loader = DataLoader(dataset=train_dataset, batch_size=1)
imgtensor, label = next(iter(single_loader))

# Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)


@timefunc
def train():
    print('Start training')
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')

        for batch_idx, (data, target) in enumerate(train_loader):
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
