# Imports
import torch.nn as nn


# Models
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
        self.fc2 = nn.Linear(in_features=self.num_in
                             * self.num_hidden, out_features=1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        bs = x.shape[0]
        x = self.bn(x)
        x = x.view(-1, 2)
        x = self.fc1(x)
        x = self.act1(x).reshape(bs, -1)
        x = self.fc2(x)
        return self.act2(x).view(-1)
