import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_data, get_data_and_features
from utils import get_predictions, get_submission


class NN(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(NN, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(input_size // 2 * hidden_dim, 1)

    def forward(self, x):
        N = x.shape[0]
        x = self.bn(x)
        orig_features = x[:, :200].unsqueeze(2)  # (N, 200, 1)
        new_features = x[:, 200:].unsqueeze(2)  # (N, 200, 1)
        x = torch.cat([orig_features, new_features], dim=2)  # (N, 200, 2)
        x = F.relu(self.fc1(x)).reshape(N, -1)  # (N, 200*hidden_dim)
        return torch.sigmoid(self.fc2(x)).view(-1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = NN(input_size=400, hidden_dim=100).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
loss_fn = nn.BCELoss()
get_data_and_features()
train_ds, val_ds, test_ds, test_ids = get_data()
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)

for epoch in tqdm(range(20)):
    probabilities, true = get_predictions(val_loader, model, device=DEVICE)
    print(f"VALIDATION ROC: {metrics.roc_auc_score(true, probabilities)}")

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward
        scores = model(data)
        loss = loss_fn(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

get_submission(model, test_loader, test_ids, DEVICE)
