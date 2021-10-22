from abc import ABC, abstractmethod


class IReporter(ABC):
    """Interface for a reporter."""

    @abstractmethod
    def notify():
        """Receive notifications from metrics."""


class ConsoleReporter(IReporter):
    """The ConsoleReporter prints the name and value of a metric
    the console."""

    def __init__(self) -> None:
        self.name = "ConsoleReporter"

    def notify(self, metric, *args):
        print(f"{metric.name}: {metric.value}")


class TensorboardReporter(IReporter):

    def __init__(self) -> None:
        self.name = "TensorboardReporter"

    def notify(self, metric, *args):
        # TODO: implement
        pass
