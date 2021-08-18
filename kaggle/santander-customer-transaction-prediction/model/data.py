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
        df_train = pd.read_csv(path)
        print(f'Imported {path}')
        path = DATAPATH + 'test.csv'
        df_test = pd.read_csv(path)
        print(f'Imported {path}')
    except FileNotFoundError:
        print(f'File not found: {path}')

    colnames = [f'var_{i}' for i in range(200)]

    # Train and validation set
    df_train_isunique = gen_isunique(df=df_train, colnames=colnames)
    df_train_hasunique = gen_hasunique(df=df_train_isunique)
    df_train = pd.concat(
        [df_train, df_train_isunique, df_train_hasunique], axis=1)

    # Test set
    df_test_isunique = gen_isunique(df=df_test, colnames=colnames)
    df_test_hasunique = gen_hasunique(df=df_test_isunique)
    df_test = pd.concat([df_test, df_test_isunique, df_test_hasunique], axis=1)

    # Tensordataset, split trainval_ds in train_ds and val_ds
    y_train = torch.tensor(
        df_train['target'].values, dtype=torch.float32).to(device)
    df_train.drop(['target', 'ID_code'], axis=1, inplace=True)
    X_train = torch.tensor(df_train.values, dtype=torch.float32)

    df_test_real = df_test.loc[df_test['has_unique'] == 1.]
    df_test_fake = df_test.loc[df_test['has_unique'] != 1.]

    test_idcode = df_test['ID_code']  # ID_code for kaggle submission
    df_test.drop('ID_code', axis=1, inplace=True)
    X_test = torch.tensor(df_test.values, dtype=torch.float32)

    trainval_ds = TensorDataset(X_train, y_train)
    train_ds, val_ds = random_split(
        dataset=trainval_ds,
        lengths=[floor(0.8 * len(trainval_ds)), ceil(0.2 * len(trainval_ds))])
    test_ds = TensorDataset(X_test)

    return train_ds, val_ds, test_ds, test_idcode


def gen_isunique(df, colnames=None):
    """Returns if a value in a column of a dataframe is
    unique (1) or not (0)

    Parameters
    ----------
    df : pandas.DataFrame
    colnames : List of columns to be checked for unique values

    Returns
    -------
    df_is_unique : pandas.DataFrame
    """
    df_is_unique = pd.DataFrame()

    if colnames is None:
        colnames = df.columns

    for col in colnames:
        count = df[col].value_counts()
        is_unique = {f"{col}_u": df[col].isin(count.index[count == 1]) * 1.}
        df_res = pd.DataFrame.from_dict(is_unique)
        df_is_unique = pd.concat([df_is_unique, df_res], axis=1)

    return df_is_unique


def gen_hasunique(df, colnames=None):
    """Returns if a row has at least one value of True or 1 over
    the columns in colnames.

    Parameters
    ----------
    df : pandas.DataFrame
    colnames : List of columns to be checked for True values

    Returns
    -------
    df_has_unique : pd.DataFrame with a single column of results"""
    if colnames is None:
        colnames = df.columns

    has_unique = {"has_unique": df[colnames].any(axis=1) * 1.}
    df_has_unique = pd.DataFrame.from_dict(has_unique)

    return df_has_unique
