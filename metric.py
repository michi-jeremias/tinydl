from abc import ABC, abstractmethod

import sklearn.metrics


class IMetric(ABC):
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


class DummyMetric(IMetric):

    def __init__(self) -> None:
        super().__init__()
        self.name = "Dummey Metric"
        self.value = -1.

    def notify(self, *args) -> None:
        """Sends the name and last value of the metric to the reporter."""
        for reporter in self._reporters:
            reporter.notify(self, *args)

    def calculate(self, scores, targets) -> None:
        self.value += 1


class RocAuc(IMetric):

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
