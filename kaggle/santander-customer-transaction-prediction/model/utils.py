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
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(
        ['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(
        ['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
