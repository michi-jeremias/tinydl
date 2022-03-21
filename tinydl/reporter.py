from abc import ABC, abstractmethod

from torch.utils.tensorboard import SummaryWriter

from tinydl.metric import Metric
from tinydl.stage import Stage


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

    def notify(self, metric: Metric, stage: Stage, *args):
        print(
            f"({self.name}) Stage: {stage.name}, Metric {metric.name}: {metric.value:.6f}")


class TensorboardScalarReporter(Reporter):

    def __init__(self,
                 name: str = None,
                 hparam: dict = {}) -> None:
        self.name = name if name else "TensorboardScalarReporter"
        self.hparam = hparam
        self.hparam_string = "_".join(
            [f"{key}_{self.hparam[key]}" for key in self.hparam])
        # self.log_dir = "runs/" + stage.name + "_" + self.hparam_string
        print(self.hparam_string)
        self.log_dir = "runs/" + "TRAIN" + "_" + self.hparam_string
        print(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.step = 0

    def notify(self, metric, stage: Stage, *args):
        self.writer.add_scalar(
            metric.name,
            scalar_value=metric.value,
            global_step=self.step
        )
        self.step += 1


class TensorboardHparamReporter(Reporter):
    """The TensorboardHparamReporter calls
    tensorboard.SummaryWriter().add_hparams() with the name and value of
    the metric. Use after the Runner() is finished!"""

    def __init__(self,
                 name: str = None,
                 hparam: dict = {}) -> None:
        self.name = name if name else "TensorboardHparamReporter"
        self.hparam = hparam
        self.hparam_string = self.name + "_".join(
            [f"{key}_{self.hparam[key]}" for key in self.hparam])
        self.log_dir = "runs/" + "TRAIN" + "_" + self.hparam_string
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def notify(self, metric_dict, stage: Stage, *args):
        self.writer.add_hparams(
            hparam_dict=self.hparam,
            metric_dict=metric_dict,
        )
