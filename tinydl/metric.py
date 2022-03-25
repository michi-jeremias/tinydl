from abc import ABC, abstractmethod

import sklearn.metrics
from torch.nn import BCELoss


class Metric(ABC):
    """Interface for a Metric. The metric gets reported through a
    Reporter."""

    def __init__(self, name: str = None) -> None:
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class BinaryCrossentropy(Metric):

    def __init__(self, name: str = None) -> None:
        super().__init__()
        self.name = "BCE" if not name else name
        self.value = -1.
        self.bce_loss = BCELoss()

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def calculate(self, scores, targets) -> None:
        self.value = self.bce_loss(scores, targets)


class DummyMetric(Metric):

    def __init__(self, name: str = None) -> None:
        super().__init__()
        self.name = "Dummy Metric" if not name else name
        self.value = 1.

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def calculate(self, scores, targets) -> None:
        self.value = self.value * 0.95


class RocAuc(Metric):

    def __init__(self, name: str = None) -> None:
        super().__init__()
        self.name = "ROC_AUC" if not name else name
        self.value = -1.

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def calculate(self, scores, targets) -> None:
        self.value = sklearn.metrics.roc_auc_score(targets, scores)
