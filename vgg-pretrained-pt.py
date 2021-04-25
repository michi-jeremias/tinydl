# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models

from timefunc import timefunc

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
DATAPATH = '../data'
PRETRAINEDPATH = '../models'
batch_size = 64
learning_rate = 0.001

# Load pretrained model

model = models.vgg16(pretrained=True)
