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

    # Train and validation set
    df_train = create_isunique(df_train)
    y_train = torch.tensor(
        df_train['target'].values, dtype=torch.float32).to(device)
    df_train.drop(['target', 'ID_code'], axis=1, inplace=True)
    X_train = torch.tensor(df_train.values, dtype=torch.float32)
    dataset = TensorDataset(X_train, y_train)

    # Split train in train and val
    train_ds, val_ds = random_split(
        dataset=dataset,
        lengths=[floor(0.8 * len(dataset)), ceil(0.2 * len(dataset))])

    # Test set
    df_test = create_isunique(df_test)
    X_test = torch.tensor(df_test.values, dtype=torch.float32)
    test_idcode = df_test['ID_code']  # ID_code for kaggle submission
    df_test.drop('ID_code', axis=1, inplace=True)
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


def get_colnames():
    return [f'var_{i}' for i in range(200)]


def split_real_fake(df):
    # Syntax: df.loc[df[conditional], [selected data]]
    df_test_real = df.loc[df["has_unique"], ["ID_code"] + get_colnames()]
    df_test_fake = df.loc[~df["has_unique"], ["ID_code"] + get_colnames()]
    return df_test_real, df_test_fake


def gen_unique(df, colnames=None):
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
