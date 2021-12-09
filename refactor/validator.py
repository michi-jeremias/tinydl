from abc import ABCMeta, abstractmethod
from typing import List

import torch
import torch.nn
import torch.utils.data.dataloader

from metric import Metric


class ValidatorTemplate(ABCMeta):

    @abstractmethod
    @staticmethod
    def validate():
        """Validates predictions against targets."""


class Validator(ValidatorTemplate):

    def __init__(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.dataloader,
        metrics: List[Metric],
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.loader = loader
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    def validate(self) -> None:
        self.model.eval()
        all_scores = []
        all_targets = []

        with torch.no_grad():

            for batch_idx, (data, targets) in enumerate(self.loader):
                current_bs = data.shape[0]
                data = data.to(self.device)
                targets = targets.to(self.device)
                scores = self.model(data)

    # def validate(self) -> None:
    #     """Reports a metric from self.metrics_fn on the validation set."""
    #     self.model.eval()
    #     predictions = []
    #     actuals = []

    #     with torch.no_grad():

    #         for x, y in self.loader.train_loader:
    #             x = x.to(self.device)
    #             y = y.to(self.device)
    #             scores = self.model(x)
    #             predictions += scores.tolist()
    #             actuals += y.tolist()
