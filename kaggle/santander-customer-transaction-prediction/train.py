# Imports
import torch
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from data import get_data
# from utils import get_predictions
from torch.utils.data import DataLoader

from utils import get_predictions


# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
BATCH_SIZE = 1024
LR = 3e-4
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4


# Model
class NN(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(num_features=input_size),
            nn.Linear(in_features=input_size, out_features=50, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=50, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


model = NN(input_size=200).to(DEVICE)


# Data
train_ds, val_ds, test_ds, test_ids = get_data()
train_loader = DataLoader(
    dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE)


# Optimizer
optimizer = optim.Adam(params=model.parameters(),
                       lr=LR, weight_decay=WEIGHT_DECAY)


# Loss
loss_fn = nn.BCELoss()


# Train
def train(num_epochs=NUM_EPOCHS):
    model.train()

    for epoch in range(num_epochs):
        probabilities, actuals = get_predictions(val_loader, model, DEVICE)
        print(
            f'Validation ROC: {metrics.roc_auc_score(actuals, probabilities)}')

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward
            scores = model(data)
            loss = loss_fn(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx == 0:
                print(loss)
