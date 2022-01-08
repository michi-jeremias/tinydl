from abc import ABC, abstractmethod

from torch.utils.tensorboard import SummaryWriter

from tinydl.metric import Metric


class Reporter(ABC):
    """Interface for a reporter."""

    @abstractmethod
    def notify():
        """Receive notifications from metrics and report them."""


class ConsoleReporter(Reporter):
    """The ConsoleReporter prints the name and value of a metric
    the console."""

    def __init__(self, name: str = None) -> None:
        self.name = name if name else "ConsoleReporter"

    def notify(self, metric, *args):
        print(f"({self.name}) {metric.name}: {metric.value}")


# class TensorboardScalarReporter(Reporter):

#     def __init__(self, name: str = None, log_dir: str = None) -> None:
#         self.name = name if name else "TensorboardScalarReporter"
#         self.log_dir = log_dir
#         self.writer = SummaryWriter(log_dir=self.log_dir)
#         self.step = 0

#     def notify(self, metric, *args):
#         self.writer.add_scalar(
#             metric.name,
#             scalar_value=metric.value,
#             global_step=self.step
#         )
#         self.step += 1


class TensorboardHparamReporter(Reporter):
    """The TensorboardHparamReporter calls
    tensorboard.SummaryWriter().add_hparams() with the name and value of
    the metric."""

    def __init__(self, name: str = None, hparam: dict = {}) -> None:
        self.name = name if name else "TensorboardHparamReporter"
        self.hparam = hparam
        dir_name = "_".join(
            [f"{key}_{self.hparam[key]}" for key in self.hparam])
        self.log_dir = "runs/" + dir_name
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.step = 0

    def notify(self, metric: Metric, *args):

        self.writer.add_scalar(
            metric.name,
            scalar_value=metric.value,
            global_step=self.step
        )
        self.writer.add_hparams(
            hparam_dict=self.hparam,
            metric_dict={f"{metric.name}": metric.value})
        self.step += 1
