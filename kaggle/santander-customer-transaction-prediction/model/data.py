"""Data generation and modification for the kaggle competition santander
customer transaction prediction.
See https://www.kaggle.com/c/santander-customer-transaction-prediction
"""


# Imports
from abc import ABC, abstractmethod
from functools import wraps
from math import ceil, floor

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Paths, constants
DATAPATH = '../data/'
LOGPATH = '../logs/'


class DataGenerator(ABC):

    @abstractmethod
    def assert_datatype(func):
        pass

    @abstractmethod
    def generate(self):
        pass


class PandasDataGenerator(DataGenerator):

    def assert_datatype(func):
        @wraps(func)
        # def wrapper(self, data, colnames):
        def wrapper(self, data, colnames=None):
            DATATYPE = pd.DataFrame
            assert isinstance(
                data, DATATYPE), f"Argument type is not {DATATYPE}"
            return func(data, colnames)
        return wrapper

    @abstractmethod
    def generate(self, data):
        pass


class IsUniqueGenerator(PandasDataGenerator):

    @PandasDataGenerator.assert_datatype
    def generate(data, colnames=None):
        """Returns if a value in a column of a dataframe is
        unique (1) or not (0)

        Example
        -------
        [[1, 0, 0],                 [[0, 0, 0],
         [1, 0, 0],    generates     [0, 0, 0],
         [1, 1, 1],                  [0, 1, 0],
         [2, 0, 1]]                  [1, 0, 0]]

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
            colnames = data.columns

        for col in colnames:
            count = data[col].value_counts()
            is_unique = {f"{col}_u": data[col].isin(
                count.index[count == 1]) * 1.}
            df_res = pd.DataFrame.from_dict(is_unique)
            df_is_unique = pd.concat([df_is_unique, df_res], axis=1)

        return df_is_unique


class HasUniqueGenerator(PandasDataGenerator):

    @PandasDataGenerator.assert_datatype
    def generate(data, colnames=None):
        """Returns if a row has at least one value of True or 1 over
        the columns in colnames. It's basically the evaluation of an OR
        statement of a row across all columns.

        Example
        -------
        [[1, 0, 0],                 [[1],
         [0, 0, 0],    generates     [0],
         [1, 1, 1],                  [1],
         [0, 0, 1]]                  [1]]

        Parameters
        ----------
        df : pandas.DataFrame
        colnames : List of columns to be checked for True values

        Returns
        -------
        df_has_unique : pd.DataFrame with a single column of results"""
        if colnames is None:
            colnames = data.columns

        has_unique = {"has_unique": data[colnames].any(axis=1) * 1.}
        df_has_unique = pd.DataFrame.from_dict(has_unique)

        return df_has_unique


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

    iug = IsUniqueGenerator()
    # hug = HasUniqueGenerator()  # Only used
    # Train and validation set
    df_train_isunique = iug.generate(data=df_train, colnames=colnames)
    # df_train_hasunique is only useful to check the legitimacy of train
    # samples.
    # df_train_hasunique = hug.generate(data=df_train_isunique)
    # df_train = pd.concat(
    # [df_train, df_train_isunique, df_train_hasunique], axis=1)
    df_train = pd.concat([df_train, df_train_isunique], axis=1)

    # Test set
    df_test_isunique = iug.generate(data=df_test, colnames=colnames)
    # df_test_hasunique is only useful to check the legitimacy of test
    # samples.
    # df_test_hasunique = hug.generate(data=df_test_isunique)
    # df_test = pd.concat(
    # [df_test, df_test_isunique, df_test_hasunique], axis=1)
    df_test = pd.concat([df_test, df_test_isunique], axis=1)

    # Tensordataset, split trainval_ds in train_ds and val_ds
    y_train = torch.tensor(
        df_train['target'].values, dtype=torch.float32).to(DEVICE)
    df_train.drop(['target', 'ID_code'], axis=1, inplace=True)
    X_train = torch.tensor(df_train.values, dtype=torch.float32)

    # Unnecessary
    # df_test_real = df_test.loc[df_test['has_unique'] == 1.]
    # df_test_fake = df_test.loc[df_test['has_unique'] != 1.]

    df_test_idcode = df_test['ID_code']  # ID_code for kaggle submission
    # df_test_real_idcode = df_test_real['ID_code']
    # df_test_fake_idcode = df_test_fake['ID_code']
    df_test.drop('ID_code', axis=1, inplace=True)
    # df_test_real.drop('ID_code', axis=1, inplace=True)
    # df_test_fake.drop('ID_code', axis=1, inplace=True)
    X_test = torch.tensor(df_test.values, dtype=torch.float32)

    trainval_ds = TensorDataset(X_train, y_train)
    train_ds, val_ds = random_split(
        dataset=trainval_ds,
        lengths=[floor(0.8 * len(trainval_ds)), ceil(0.2 * len(trainval_ds))])
    test_ds = TensorDataset(X_test)

    return train_ds, val_ds, test_ds, df_test_idcode
