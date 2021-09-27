from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import datetime


class Writer(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def write():
        pass


class SantanderWriter(Writer):
    """Generates the output for the submission file of
    the santander transaction prediction competition at kaggle:
    https://www.kaggle.com/c/santander-customer-transaction-prediction/submit.
    """

    def __init__(self, model, test_loader, test_ids) -> None:
        super().__init__()
        self.model = model
        self.loader = test_loader
        self.test_ids = test_ids

    def write(self):
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for x, _ in tqdm(self.loader):
                x = x.to(self.device)
                score = self.model(x)
                prediction = score.float()
                all_preds += prediction.tolist()

        self.model.train()

        df = pd.DataFrame({
            "ID_code": self.test_ids.values,
            "target": np.array(all_preds)
        })

        now = datetime.datetime.now()
        date = f"{now.year}{now.month:02d}{now.day:02d}"
        time = f"{now.hour}{now.minute:02d}{now.second:02d}"
        filename = f"{date}_{time}"
        df.to_csv("/files" + filename, index=False)
