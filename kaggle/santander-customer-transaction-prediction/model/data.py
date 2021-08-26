"""Data generation and modification for the kaggle competition santander
customer transaction prediction.
See https://www.kaggle.com/c/santander-customer-transaction-prediction
"""

# Imports
from math import ceil, floor

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split

from datagenerator import HasUniqueGenerator, IsUniqueGenerator


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

    iug = IsUniqueGenerator()
    # hug = HasUniqueGenerator()  # Used to find real/fake data in test set
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
        lengths=[floor(0.999 * len(trainval_ds)),
                 ceil(0.001 * len(trainval_ds))])
    # lengths=[floor(0.8 * len(trainval_ds)), ceil(0.2 * len(trainval_ds))])
    test_ds = TensorDataset(X_test, y_train)

    return train_ds, val_ds, test_ds, df_test_idcode


def get_submission(model, loader, test_ids, device):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            # print(x.shape)
            x = x.to(device)
            score = model(x)
            prediction = score.float()
            all_preds += prediction.tolist()

    model.train()

    df = pd.DataFrame({
        "ID_code": test_ids.values,
        "target": np.array(all_preds)
    })

    df.to_csv(DATAPATH + "sub.csv", index=False)
