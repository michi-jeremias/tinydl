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

from data.datagenerator import HasUniqueGenerator, IsUniqueGenerator
from data.const import DATAPATH, LOGPATH


# Augment data
def prepare_data():
    # Read csv data, drop target column (train.csv) and ID_code column
    # (train.csv, test.csv).
    # Split train into train and validation.
    # Return TensorDatasets for train, val and test. Return ID_Code for
    # the test set.

    # Read pandas dataframes
    df_train, df_test = None, None

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
    hug = HasUniqueGenerator()  # Used to find real/fake data in test set

    # Generate unique information in test set
    df_test_isunique = iug.generate(df_test, colnames=colnames)
    df_test_hasunique = hug.generate(df_test_isunique)

    df_test_all = pd.concat(
        [df_test, df_test_isunique, df_test_hasunique], axis=1)

    df_test_real = df_test_all.loc[df_test_all['has_unique'] == 1, [
        "ID_code"] + colnames]
    df_test_fake = df_test_all.loc[df_test_all['has_unique'] != 1, [
        "ID_code"] + colnames]

    df_train_test = pd.concat([df_train, df_test_real], axis=0)
    df_isunique = iug.generate(df_train_test, colnames=colnames)
    df_train_test_isunique = pd.concat([df_train_test, df_isunique], axis=1)
    df_zeros = pd.DataFrame(
        np.zeros_like(df_test_fake[colnames], dtype=np.short),
        columns=[col + "_u" for col in colnames],
        index=df_test_fake.index)
    df_test_fake_isunique = pd.concat([df_test_fake, df_zeros], axis=1)

    df_train = df_train_test_isunique[
        df_train_test_isunique["ID_code"].str.contains("train")].copy()
    df_test_real_isunique = df_train_test_isunique[
        df_train_test_isunique["ID_code"].str.contains("test")].copy().drop([
            "target"], axis=1)
    df_test = pd.concat([df_test_real_isunique, df_test_fake_isunique], axis=0)

    df_train.to_csv(DATAPATH + "aug_train.csv", index=False)
    df_test.to_csv(DATAPATH + "aug_test.csv", index=False)
    # Train and validation set


def get_data(train="aug_train.csv", test="aug_test.csv", submission=True):
    train_data = pd.read_csv(DATAPATH + train)
    y = train_data["target"]
    X = train_data.drop(["ID_code", "target"], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)

    if submission:
        train_ds, val_ds = random_split(
            ds, [int(0.999 * len(ds)), ceil(0.001 * len(ds))])
    else:
        train_ds, val_ds = random_split(
            ds, [int(0.8 * len(ds)), ceil(0.2 * len(ds))])

    test_data = pd.read_csv(DATAPATH + test)
    test_ids = test_data["ID_code"]
    X = test_data.drop(["ID_code"], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    test_ds = TensorDataset(X_tensor, y_tensor)

    return train_ds, val_ds, test_ds, test_ids


def get_submission(model, loader, test_ids, device, filename="sub.csv"):
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

    df.to_csv(DATAPATH + filename, index=False)
