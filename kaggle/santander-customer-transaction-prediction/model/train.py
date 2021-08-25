# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_data
from model import SimpleNet, OneDimNet, TwoDimNet, NN
from utils import get_predictions, plot_correlations

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
BATCH_SIZE = 1024
LR = 3e-4
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4


model = SimpleNet(num_in=200, num_hidden=16).to(DEVICE)
model = OneDimNet(num_in=200, num_hidden=16).to(DEVICE)
model = TwoDimNet(num_in=400, num_hidden=100).to(DEVICE)
model = NN(input_size=400, hidden_dim=100).to(DEVICE)


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
    print("Start training.")
    model.train()

    for epoch in range(num_epochs):
        probabilities, actuals = get_predictions(val_loader, model, DEVICE)
        print(
            f"[{epoch + 1}/{num_epochs}]/ Validation ROC: "
            f"{metrics.roc_auc_score(actuals, probabilities)}")

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

    print("Finished training.")
