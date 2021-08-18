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
    """Plots correlations of data.

    Parameters
    ----------
    df : pandas.DataFrame"""
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
