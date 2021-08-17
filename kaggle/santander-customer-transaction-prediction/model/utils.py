# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# Functions
def get_predictions(loader, model, device):
    model = model.to(device)
    model.eval()
    saved_predictions = []
    true_labels = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)

            saved_predictions += scores.tolist()
            true_labels += targets.tolist()

    model.train()
    return saved_predictions, true_labels


def plot_correlations(df):
    # df is a pandas dataframe
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(
        ticks=range(0, df.select_dtypes(['number']).shape[1], 10),
        fontsize=14,
        rotation=0)
    plt.yticks(
        ticks=range(0, df.select_dtypes(['number']).shape[1], 10),
        fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=18)


def gen_testtensor():
    """Generates a 4x4 tensor for testing purposes"""
    return torch.tensor([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]
    ])


def gen_testdf():
    """Generates a 4x4 dataframe for testing purposes

    returns
    """
    return pd.DataFrame([
        [1, 2, 3, 4],
        [2, 2, 3, 4],
        [2, 2, 3, 5],
        [2, 2, 3, 4]
    ])


colnames = range(4)


def gen_unique(df, colnames=None):
    """Returns if a value in a column of a dataframe is
    unique (1) or not (0)

    Parameters
    ----------
    df : pandas.DataFrame
    colnames : list of columns to be checked for unique values

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


def gen_hasunique(df, colnames):
    """"""
    has_unique = {"has_unique": df[colnames].any(axis=1)}
    df_has_unique = pd.DataFrame.from_dict(has_unique)
