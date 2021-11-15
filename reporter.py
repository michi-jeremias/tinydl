from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter


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
        print(f"({self.name}) {metric.name}: {metric.value}")


class TensorboardScalarReporter(Reporter):

    def __init__(self, name=None, log_dir=None) -> None:
        self.name = name if name else "TensorboardScalarReporter"
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def notify(self, metric, *args):
        print("notify tb logger")
        self.writer.add_scalar(
            metric.name,
            scalar_value=metric.value,
            global_step=self.step
        )
        self.step += 1


class TensorboardHparamReporter(Reporter):

    def __init__(self, name=None, log_dir=None) -> None:
        self.name = name if name else "TensorboardHparamReporter"
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def notify(self, metric, hparam: dict, *args):
        print("notify tb logger")
        self.writer.add_hparams(
            # hparam_dict={'learningrate': lr, 'batchsize': bs},
            hparam_dict=hparam,
            metric_dict={f"{metric.name}": metric.value)

        self.writer.add_scalar(
            metric.name,
            scalar_value=metric.value,
            global_step=self.step
        )
        self.step += 1
