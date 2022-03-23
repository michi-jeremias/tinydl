from abc import ABC, abstractmethod

from torch.utils.tensorboard import SummaryWriter

from tinydl.metric import Metric
from tinydl.stage import Stage


class Reporter(ABC):
    """Interface for a reporter."""

    @abstractmethod
    def report():
        """Receive values from metrics and report them."""

    def add_metrics(self, metrics) -> None:
        """Add a Metric().

        Parameters
        ----------
        metrics : Metric() or [Metric()]"""

        metrics = metrics if isinstance(
            metrics, list) else [metrics]
        for metric in metrics:
            self._metrics.add(metric)

    def remove_metric(self, metric) -> None:
        """Remove a Metric().

        Parameters
        ----------
        metric : Metric() """
        self._metrics.remove(metric)


class ConsoleReporter(Reporter):
    """The ConsoleReporter prints the name and value of a metric
    the console."""

    def __init__(self,
                 name: str = None) -> None:
        self.name = name if name else "ConsoleReporter"
        self._metrics = set()

    def report(self, stage: Stage, scores, targets, *args):

        for metric in self._metrics:
            metric.calculate(scores, targets)
            print(
                f"({self.name}) Stage: {stage.name}, Metric {metric.name}: {metric.value:.6f}")


class TensorboardScalarReporter(Reporter):

    def __init__(self,
                 name: str = None,
                 hparam: dict = {}) -> None:
        self.name = name if name else "TensorboardScalarReporter"
        self.hparam = hparam
        self._metrics = set()
        self.hparam_string = "_".join(
            [f"{key}_{self.hparam[key]}" for key in self.hparam])
        self.writer = SummaryWriter(comment=f"_{self.hparam_string}")
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


class TensorboardHparamReporter(Reporter):
    """The TensorboardHparamReporter calls
    tensorboard.SummaryWriter().add_hparams() with the name and value of
    the metric. Use after the Runner() is finished!"""

    def __init__(self,
                 name: str = None,
                 hparam: dict = {}) -> None:
        self.name = name if name else "TensorboardHparamReporter"
        self.hparam = hparam
        self._metrics = set()
        self.hparam_string = "_".join(
            [f"{key}_{self.hparam[key]}" for key in self.hparam])
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
