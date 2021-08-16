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

    # Read pandas dataframes
    try:
        path = DATAPATH + 'train.csv'
        train_data = pd.read_csv(path)
        print(f'Imported {path}')
        path = DATAPATH + 'test.csv'
        test_data = pd.read_csv(path)
        print(f'Imported {path}')
    except FileNotFoundError:
        print(f'File not found: {path}')

    # Train and validation set
    train_data = create_isunique(train_data)
    y_train = torch.tensor(
        train_data['target'].values, dtype=torch.float32).to(device)
    train_data.drop(['target', 'ID_code'], axis=1, inplace=True)
    X_train = torch.tensor(train_data.values, dtype=torch.float32)
    dataset = TensorDataset(X_train, y_train)
    train_ds, val_ds = random_split(
        dataset=dataset,
        lengths=[floor(0.8 * len(dataset)), ceil(0.2 * len(dataset))])

    # Test set
    test_data = create_isunique(test_data)
    X_test = torch.tensor(test_data.values, dtype=torch.float32)
    test_idcode = test_data['ID_code']  # ID_code for kaggle submission
    test_data.drop('ID_code', axis=1, inplace=True)
    test_ds = TensorDataset(X_test)

    return train_ds, val_ds, test_ds, test_idcode


def create_isunique(df):
    # Creates a column for every column, telling if a value is unique
    # in that column.
    col_names = get_colnames()
    for col in col_names:
        counts = df[col].value_counts()
        uniques = counts.index[counts == 1]
        res = pd.DataFrame(df[col].isin(uniques))
        res.columns = [col + '_u']
        df = pd.concat([df, res], axis=1)
        # df[col + '_unique'] = df[col].isin(uniques)

    # The next line creates the information if an observation contains
    # a unique value at all. If not, the data is considered to be fake
    # (generated).
    df['has_unique'] = df[[col + '_u' for col in col_names]].any(axis=1)
    return df


def create_isunique2(df):
    # Creates a column for every column, telling if a value is unique
    # in that column.
    col_names = [f'var_{i}' for i in range(200)]
    for col in col_names:
        counts = df[col].value_counts()
        uniques = counts.index[counts == 1]
        df[col + '_u'] = df[col].isin(uniques)
        # df[col + '_unique'] = df[col].isin(uniques)
    return df


def get_colnames():
    return [f'var_{i}' for i in range(200)]


def split_real_fake(df):
    # Syntax: df.loc[df[conditional], [selected data]]
    real_test = df.loc[df["has_unique"], ["ID_code"] + get_colnames()]
    fake_test = df.loc[~df["has_unique"], ["ID_code"] + get_colnames()]
    return real_test, fake_test
