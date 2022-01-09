from abc import ABC, abstractmethod

import sklearn.metrics
# from torch.nn import BCELoss
from torch.nn import BCELoss


class Metric(ABC):
    """Interface for a Metric. The metric gets reported through a
    Reporter."""

    def __init__(self, name: str = None) -> None:
        self._reporters = set()
        self.name = name

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

    def __init__(self, name: str = None) -> None:
        super().__init__()
        self.name = "Dummy Metric" if not name else name
        self.value = 1.

    def notify(self, *args) -> None:
        """Sends the name and last value of the metric to the reporter."""
        for reporter in self._reporters:
            reporter.notify(self, *args)

    def calculate(self, scores, targets) -> None:
        self.value = self.value * 0.95


class RocAuc(Metric):

    def __init__(self, name: str = None) -> None:
        super().__init__()
        self.name = "ROC_AUC" if not name else name
        self.value = -1.

    def notify(self, *args) -> None:
        """Sends the name and last value of the metric to the reporter."""
        for reporter in self._reporters:
            reporter.notify(self, *args)

    def calculate(self, scores, targets) -> None:
        self.value = sklearn.metrics.roc_auc_score(targets, scores)


class BinaryCrossentropy(Metric):

    def __init__(self, name: str = None) -> None:
        super().__init__()
        self.name = "BCE" if not name else name
        self.value = -1.
        self.bce_loss = BCELoss()

    def notify(self, *args) -> None:
        """Sends the name and last value of the metric to the reporter."""
        for reporter in self._reporters:
            reporter.notify(self, *args)

    def calculate(self, scores, targets) -> None:
        self.value = self.bce_loss(scores, targets)
