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
