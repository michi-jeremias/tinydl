from abc import ABC, abstractmethod
from types import list

from torch.utils.tensorboard import SummaryWriter

from tinydl.metric import Metric
from tinydl.report import Report
from tinydl.stage import Stage


class Reporter(ABC):
    """Interface for a reporter."""

    def __init__(self) -> None:
        self.name = "Reporter"
        self._metrics = set()
        self.reports = []

    def __repr__(self) -> str:
        return f"{__class__.__name__}(name: {self.name}, metrics: {self._metrics})"

    @abstractmethod
    def report():
        """Receive values from metrics and report them.
        Must be implemented in a Reporter."""

    def calculate_metrics(self, stage: Stage, scores, targets, *args):
        """Triggers Metric.calculate()."""
        try:
            for metric in self._metrics:
                self.reports.append(
                    Report(
                        metric_name=metric.name,
                        metric_value=metric.calculate(scores, targets),
                        stage=stage,
                    )
                )

        except Exception as e:
            print(f"Error in {self.__class__}")
            print(e)

    def add_metrics(self, metrics: list[Metric]) -> None:
        """Add a Metric().

        Parameters
        ----------
        metrics : Metric() or [Metric()]"""

        metrics = metrics if isinstance(
            metrics, list) else [metrics]
        for metric in metrics:
            self._metrics.add(metric)

    def remove_metrics(self, metrics: list[Metric]) -> None:
        metrics = metrics if isinstance(metrics, list) else [metrics]

        for metric in metrics:

            try:
                self._metrics.discard(metric)
            except Exception as e:
                print(e)

    def flush_metrics(self) -> None:
        """Remove all metrics."""

        self._metrics = set()


class ConsoleReporter(Reporter):
    """The ConsoleReporter prints the name and value of a metric
    the console."""

    def __init__(self,
                 name: str = None) -> None:
        super().__init__()
        self.name = name if name else "ConsoleReporter"

    def report(self):
        for report in self.reports:
            print(
                f"({self.name}) stage: {report.stage}, metric: {report.metric_name}, value: {report.metric_value:.6f}")


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

    def report(self):
        for report in self.reports:
            self.writer.add_scalar(
                tag=report.metric_name + "_" + report.stage,
                scalar_value=report.metric_value,
                global_step=self.step
            )
            self.step += 1


class TensorboardHparamReporter(Reporter):
    """The TensorboardHparamReporter calls
    tensorboard.SummaryWriter().add_hparams() with the name and value of
    the metric. Note that all metrics in writer.add_hparams() have to be
    written at once, otherwise there might be problems with tensorboard
    identifying metric names as intended."""

    def __init__(self,
                 name: str = None,
                 hparam: dict = {}) -> None:
        self.name = name if name else "TensorboardHparamReporter"
        self.hparam = hparam
        self._metrics = set()
        self.hparam_string = "_".join(
            [f"{key}_{self.hparam[key]}" for key in self.hparam])
        self.writer = SummaryWriter(comment=f"_{self.hparam_string}")
        self.metric_dict = {}

    def report(self):

        for report in self.reports:
            self.metric_dict[f"{report.metric_name}_{report.stage}"] = report.metric_value.item(
            )

        self.writer.add_hparams(
            hparam_dict=self.hparam,
            metric_dict=self.metric_dict,
        )
        self.metric_dict = {}
