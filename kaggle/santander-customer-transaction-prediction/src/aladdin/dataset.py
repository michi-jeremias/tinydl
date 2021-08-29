from math import ceil

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from tqdm import tqdm


DATAPATH = "../../data/"


def get_data_and_features():
    # execute before get_data()
    train = pd.read_csv(DATAPATH + "train.csv")
    test = pd.read_csv(DATAPATH + "test.csv")

    col_names = [f"var_{i}" for i in range(200)]
    for col in tqdm(col_names):
        count = test[col].value_counts()
        uniques = count.index[count == 1]
        test[col + "_u"] = test[col].isin(uniques)

    for col in tqdm(col_names):
        count = train[col].value_counts()
        uniques = count.index[count == 1]
        train[col + "_u"] = test[col].isin(uniques)

    test["has_unique"] = test[[col + "_u" for col in col_names]].any(axis=1)

    real_test = test.loc[test["has_unique"], ["ID_code"] + col_names]
    fake_test = test.loc[~test["has_unique"], ["ID_code"] + col_names]
    train_and_test = pd.concat([train, real_test], axis=0)

    for col in tqdm(col_names):
        count = train_and_test[col].value_counts().to_dict()
        train_and_test[col + "_unique"] = train_and_test[col].apply(
            lambda x: 1 if count[x] == 1 else 0).values
    fake_test[col + "_unique"] = 0

    real_test = train_and_test[train_and_test["ID_code"].str.contains(
        "test")].copy()
    real_test.drop(["target"], axis=1, inplace=True)
    train = train_and_test[train_and_test["ID_code"].str.contains(
        "train")].copy()

    test = pd.concat([real_test, fake_test], axis=0)

    train.to_csv(DATAPATH + "new_shiny_train.csv", index=False)
    test.to_csv(DATAPATH + "new_shiny_test.csv", index=False)


def get_shiny_data():
    train_data = pd.read_csv(DATAPATH + "new_shiny_train.csv")
    y = train_data["target"]
    X = train_data.drop(["ID_code", "target"], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    train_ds, val_ds = random_split(
        ds, [int(0.999 * len(ds)), ceil(0.001 * len(ds))])

    test_data = pd.read_csv(DATAPATH + "new_shiny_test.csv")
    print(f"Test shape: {test_data.shape}")
    test_ids = test_data["ID_code"]
    X = test_data.drop(["ID_code"], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    test_ds = TensorDataset(X_tensor, y_tensor)

    return train_ds, val_ds, test_ds, test_ids


def get_data():
    train_data = pd.read_csv(DATAPATH + "train.csv")
    y = train_data["target"]
    X = train_data.drop(["ID_code", "target"], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    train_ds, val_ds = random_split(
        ds, [int(0.8 * len(ds)), ceil(0.2 * len(ds))])

    test_data = pd.read_csv(DATAPATH + "test.csv")
    test_ids = test_data["ID_code"]
    X = test_data.drop(["ID_code"], axis=1)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    test_ds = TensorDataset(X_tensor, y_tensor)

    return train_ds, val_ds, test_ds, test_ids
