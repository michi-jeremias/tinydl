from abc import ABC, abstractmethod

from torch.utils.tensorboard import SummaryWriter

from tinydl.metric2 import Metric2
from tinydl.stage import Stage


class Reporter2(ABC):
    """Interface for a reporter."""

    @abstractmethod
    def report():
        """Receive values from metrics and report them."""

    def subscribe(self, reporter) -> None:
        """Subscribe to a Metric().

        Parameters
        ----------
        reporter : Reporter() """

        self._metrics.add(reporter)

    def unsubscribe(self, reporter) -> None:
        """Unsubscribe from a Metric().

        Parameters
        ----------
        reporter : Reporter() """
        self._metrics.remove(reporter)


class ConsoleReporter2(Reporter2):
    """The ConsoleReporter prints the name and value of a metric
    the console."""

    def __init__(self,
                 name: str = None,
                 stage: Stage = Stage.UNDEFINED) -> None:
        self.name = name if name else "ConsoleReporter"
        self.stage = stage
        self._metrics = set()

    def report(self, stage: Stage, scores, targets, *args):

        for metric in self._metrics:
            metric.calculate(scores, targets)
            print(
                f"({self.name}) Stage: {stage.name}, Metric {metric.name}: {metric.value:.6f}")


class TensorboardScalarReporter2(Reporter2):

    def __init__(self,
                 name: str = None,
                 stage: Stage = Stage.UNDEFINED,
                 hparam: dict = {}) -> None:
        self.name = name if name else "TensorboardScalarReporter"
        self.stage = stage
        self.hparam = hparam
        self._metrics = set()

        self.hparam_string = "_".join(
            [f"{key}_{self.hparam[key]}" for key in self.hparam])

        # self.log_dir = "runs/" + self.stage.name + "_" + self.hparam_string
        # print(f"logdir: {self.log_dir}")
        self.writer = SummaryWriter(comment=f"_{self.hparam_string}")
        # self.writer = SummaryWriter(log_dir=self.log_dir)

        self.step = 0

    def report(self, stage: Stage, scores, targets, *args):

        for metric in self._metrics:
            metric.calculate(scores, targets)
            self.writer.add_scalar(
                metric.name + "_" + stage.name,
                scalar_value=metric.value,
                global_step=self.step
            )
        self.step += 1


class TensorboardHparamReporter2(Reporter2):
    """The TensorboardHparamReporter calls
    tensorboard.SummaryWriter().add_hparams() with the name and value of
    the metric. Use after the Runner() is finished!"""

    def __init__(self,
                 name: str = None,
                 stage: Stage = Stage.UNDEFINED,
                 hparam: dict = {}) -> None:
        self.name = name if name else "TensorboardHparamReporter"
        self.stage = stage
        self.hparam = hparam
        self._metrics = set()
        self.hparam_string = "_".join(
            [f"{key}_{self.hparam[key]}" for key in self.hparam])
        self.log_dir = "runs/" + self.stage.name + "_" + self.hparam_string
        self.writer = SummaryWriter(comment=f"_{self.hparam_string}")

    def report(self, stage: Stage, scores, targets, *args):
        metric_dict = {}

        try:
            for metric in self._metrics:
                metric.calculate(scores, targets)
                metric_dict[f"{metric.name}_{stage.name}"] = metric.value.item()

            self.writer.add_hparams(
                hparam_dict=self.hparam,
                metric_dict=metric_dict,
            )
        except:
            print("no metric in self._metrics?")
