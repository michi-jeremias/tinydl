from abc import ABC, abstractmethod

import sklearn.metrics
# from torch.nn import BCELoss
from torch.nn import BCELoss
import torch


class Metric(ABC):
    """Interface for a Metric. The metric gets reported through a
    Reporter."""

    def __init__(self) -> None:
        self._reporters = set()

    def subscribe(self, reporter) -> None:
        """Subscribe to a Reporter().

        Parameters
        ----------
        reporter : IReporter() """

        self._reporters.add(reporter)

    def unsubscribe(self, reporter) -> None:
        """Unsubscribe from a Reporter().

        Parameters
        ----------
        reporter : IReporter() """
        self._reporters.remove(reporter)

    @abstractmethod
    def notify():
        """Push a message to the Reporter()."""


class DummyMetric(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.name = "Dummy Metric"
        self.value = 1.

    def notify(self, *args) -> None:
        """Sends the name and last value of the metric to the reporter."""
        for reporter in self._reporters:
            reporter.notify(self, *args)

    def calculate(self, scores, targets) -> None:
        self.value = self.value * 0.95


class RocAuc(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.name = "ROC_AUC"
        self.value = -1.

    def notify(self, *args) -> None:
        """Sends the name and last value of the metric to the reporter."""
        for reporter in self._reporters:
            reporter.notify(self, *args)

    def calculate(self, scores, targets) -> None:
        self.value = sklearn.metrics.roc_auc_score(targets, scores)


class BinaryCrossentropy(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.name = "BCE"
        self.value = -1.
        self.bce_loss = BCELoss()

    def notify(self, *args) -> None:
        """Sends the name and last value of the metric to the reporter."""
        for reporter in self._reporters:
            reporter.notify(self, *args)

    def calculate(self, scores, targets) -> None:
        self.value = self.bce_loss(scores, targets)
