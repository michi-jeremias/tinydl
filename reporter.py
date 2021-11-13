from abc import ABC, abstractmethod


class Reporter(ABC):
    """Interface for a reporter."""

    @abstractmethod
    def notify():
        """Receive notifications from metrics."""


class ConsoleReporter(Reporter):
    """The ConsoleReporter prints the name and value of a metric
    the console."""

    def __init__(self, name=None) -> None:
        self.name = name if name else "ConsoleReporter"

    def notify(self, metric, *args):
        print(f"{metric.name}: {metric.value}")


class TensorboardReporter(Reporter):

    def __init__(self, name=None) -> None:
        self.name = name if name else "TensorboardReporter"

    def notify(self, metric, *args):
        # TODO: implement
        pass
