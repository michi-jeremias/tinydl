# Imports
import torch
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from data import get_data
# from utils import get_predictions
from torch.utils.data import DataLoader


# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model
class NN(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(num_features=input_size),
            nn.Linear(in_features=input_size, out_features=50, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=50, out_features=1)
        )

    def forward(self, x):
        return self.net(x)


model = NN(input_size=200).to(DEVICE)


# Data
train_ds, val_ds, test_ds, test_ids = get_data()
