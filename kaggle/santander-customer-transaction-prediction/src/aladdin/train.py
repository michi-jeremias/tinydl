import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_data, get_data_and_features
from utils import get_predictions, get_submission


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.BCELoss()
train_ds, val_ds, test_ds, test_ids = get_data()
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)


# Baseline model
class Baseline(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).view(-1)


model = Baseline(200).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
# get_data_and_features()


# Improvement 1
class NN1(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(NN1, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(input_size * hidden_dim, 1)

    def forward(self, x):
        BATCH_SIZE = x.shape[0]
        x = self.bn(x)
        x = x.view(-1, 1)
        x = F.relu(self.fc1(x)).reshape(BATCH_SIZE, -1)  # (BS, input*hidden)
        return torch.sigmoid(self.fc2(x)).view(-1)


model = NN1(200, 16)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Train
NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    probabilities, true = get_predictions(val_loader, model, device=DEVICE)
    print(f"[{epoch + 1}/{NUM_EPOCHS}]VALIDATION ROC: {metrics.roc_auc_score(true, probabilities)}")
    # data, targets = next(iter(train_loader))
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward
        scores = model(data)
        loss = loss_fn(scores, targets)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Submit
get_submission(model, test_loader, test_ids, DEVICE)
