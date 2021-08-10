# Imports
import pandas as pd
import numpy as np
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
