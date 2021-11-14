# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from deeplearning.modelinit import init_normal, init_xavier


class Model():

    def __init__(self, module) -> None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.module = module.to(device)

    def init_normal(self):
        self.module.apply(init_normal)

    def init_xavier(self):
        self.module.apply(init_xavier)

    def forward(self):
        return self.module

    __call__ = forward


# NN Architectures
class SimpleNet(nn.Module):

    def __init__(self, num_in, num_hidden):
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.net = nn.Sequential(
            nn.BatchNorm1d(num_features=self.num_in),
            nn.Linear(in_features=self.num_in,
                      out_features=self.num_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.num_hidden, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


class OneDimNet(nn.Module):

    def __init__(self, num_in, num_hidden):
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden

        self.bn = nn.BatchNorm1d(num_features=self.num_in)
        self.fc1 = nn.Linear(
            in_features=1, out_features=self.num_hidden, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=self.num_in
                             * self.num_hidden, out_features=1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        bs = x.shape[0]
        x = self.bn(x)
        x = x.view(-1, 1)
        x = self.fc1(x)
        x = self.act1(x).reshape(bs, -1)
        x = self.fc2(x)
        return self.act2(x).view(-1)


class TwoDimNet(nn.Module):

    def __init__(self, num_in, num_hidden):
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden

        self.bn = nn.BatchNorm1d(num_features=self.num_in)
        self.fc1 = nn.Linear(
            in_features=2, out_features=self.num_hidden, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=self.num_in // 2 * self.num_hidden,
                             out_features=1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        bs = x.shape[0]
        x = self.bn(x)
        var = x[:, :200].unsqueeze(2)
        isunique_features = x[:, 200:].unsqueeze(2)
        x = torch.cat([var, isunique_features], dim=2)
        x = self.fc1(x)
        x = self.act1(x).reshape(bs, -1)
        x = self.fc2(x)
        return self.act2(x).view(-1)


class NN2(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(2, hidden_dim)  # changed
        self.fc2 = nn.Linear(input_size // 2 * hidden_dim, 1)

    def forward(self, x):
        BS = x.shape[0]  # Batch size
        x = self.bn(x)
        orig_features = x[:, :200].unsqueeze(2)  # (BS, 200, 1)
        new_features = x[:, 200:].unsqueeze(2)  # (BS, 200, 1)

        x = torch.cat([orig_features, new_features], dim=2)  # (BS, 200, 2)
        x = F.relu(self.fc1(x)).reshape(BS, -1)  # (N, 200 * hidden)
        return torch.sigmoid(self.fc2(x)).view(-1)


class NN3(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(2, hidden_dim)  # changed
        self.fc2 = nn.Linear(input_size // 2 * hidden_dim, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        BS = x.shape[0]  # Batch size
        x = self.bn(x)
        orig_features = x[:, :200].unsqueeze(2)  # (BS, 200, 1)
        new_features = x[:, 200:].unsqueeze(2)  # (BS, 200, 1)

        x = torch.cat([orig_features, new_features], dim=2)  # (BS, 200, 2)
        x = F.relu(self.fc1(x)).reshape(BS, -1)  # (N, 200 * hidden)
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)).view(-1)
