# Imports
import pandas as pd
import torch
from math import ceil, floor
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Paths, constants
DATAPATH = '../../data/santander-customer-transaction-prediction/'
LOGPATH = '../../logs/santander-customer-transaction-prediction/'


# Load data
def load_data():
    # Read csv data, drop target column (train.csv) and ID_code column
    # (train.csv, test.csv).
    # Split train into train and validation.
    # Return TensorDatasets for train, val and test.
    train_data = pd.read_csv(DATAPATH + 'train.csv')
    y_train = torch.tensor(train_data['target'].values, dtype=torch.float32)
    train_data.drop(['target', 'ID_code'], axis=1, inplace=True)
    X_train = torch.tensor(train_data.values, dtype=torch.float32)
    dataset = TensorDataset(X_train, y_train)
    train_ds, val_ds = random_split(
        dataset=dataset,
        lengths=[floor(0.8 * len(dataset)), ceil(0.2 * len(dataset))])

    test_data = pd.read_csv(DATAPATH + 'test.csv')
    test_idcode = test_data['ID_code']
    test_data.drop('ID_code', axis=1, inplace=True)
    X_test = torch.tensor(test_data.values, dtype=torch.float32)
    test_ds = TensorDataset(X_test)

    return train_ds, val_ds, test_ds
