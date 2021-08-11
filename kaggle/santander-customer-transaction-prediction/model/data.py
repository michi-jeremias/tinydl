"""Data loading for the kaggle competition santander customer
transaction prediction.
See https://www.kaggle.com/c/santander-customer-transaction-prediction
"""


# Imports
import pandas as pd
import torch
from math import ceil, floor
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split


# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Paths, constants
DATAPATH = '../data/'
LOGPATH = '../logs/'


# Load data
def get_data(device=DEVICE):
    # Read csv data, drop target column (train.csv) and ID_code column
    # (train.csv, test.csv).
    # Split train into train and validation.
    # Return TensorDatasets for train, val and test. Return ID_Code for
    # the test set.
    try:
        train_data = pd.read_csv(DATAPATH + 'train.csv')
        print(f'Imported {DATAPATH}train.csv')
    except FileNotFoundError:
        print(f'File not found: {DATAPATH}train.csv')
    y_train = torch.tensor(
        train_data['target'].values, dtype=torch.float32).to(device)
    train_data.drop(['target', 'ID_code'], axis=1, inplace=True)
    X_train = torch.tensor(train_data.values, dtype=torch.float32)
    dataset = TensorDataset(X_train, y_train)
    train_ds, val_ds = random_split(
        dataset=dataset,
        lengths=[floor(0.8 * len(dataset)), ceil(0.2 * len(dataset))])

    try:
        test_data = pd.read_csv(DATAPATH + 'test.csv')
        print(f'Imported {DATAPATH}test.csv')
    except FileNotFoundError:
        print(f'File not found: {DATAPATH}test.csv')

    test_idcode = test_data['ID_code']
    test_data.drop('ID_code', axis=1, inplace=True)
    X_test = torch.tensor(test_data.values, dtype=torch.float32)
    test_ds = TensorDataset(X_test)

    return train_ds, val_ds, test_ds, test_idcode
