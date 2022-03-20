from abc import ABC, abstractmethod
from typing import List

import torch
from tqdm import tqdm

from tinydl.metric import Metric
from tinydl.stage import Stage


class RunnerMediator(ABC):

    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def train():
        """To be implemented by a Trainer()"""

    @abstractmethod
    def validate():
        """To be implemented by a Validator()"""


class Trainer(RunnerMediator):

    def __init__(self,
                 loader: torch.utils.data.DataLoader,
                 optimizer,
                 loss_fn,
                 batch_metrics: List[Metric] = [],
                 epoch_metrics: List[Metric] = [],
                 ) -> None:
        super().__init__()
        self.loader = loader
        self.batch_metrics = batch_metrics if isinstance(
            batch_metrics, list) else [batch_metrics]
        self.epoch_metrics = epoch_metrics if isinstance(
            epoch_metrics, list) else [epoch_metrics]
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.stage = Stage.TRAIN

        self.metric_values = {metric.name: metric.value
                              for metric in self.batch_metrics + self.epoch_metrics}

    def train(self,
              model: torch.nn.Module) -> None:

        model.train()

        for batch_idx, (data, targets) in tqdm(enumerate(self.loader)):
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Forward
            scores = model(data)
            self.loss = self.loss_fn(scores, targets)

            # Backward
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                if self.batch_metrics:
                    for batch_metric in self.batch_metrics:
                        batch_metric.calculate(scores, targets)
                        batch_metric.notify(self.stage)

        with torch.no_grad():
            if self.epoch_metrics:
                for epoch_metric in self.epoch_metrics:
                    epoch_metric.calculate(scores, targets)
                    epoch_metric.notify(self.stage)

    def validate() -> None:
        pass


class Validator(RunnerMediator):

    def __init__(self,
                 loader: torch.utils.data.DataLoader,
                 batch_metrics: List[Metric] = [],
                 epoch_metrics: List[Metric] = [],
                 ) -> None:

        super().__init__()
        self.loader = loader
        self.batch_metrics = batch_metrics if isinstance(
            batch_metrics, list) else [batch_metrics]
        self.epoch_metrics = epoch_metrics if isinstance(
            epoch_metrics, list) else [epoch_metrics]
        self.stage = Stage.VALIDATION

    def train() -> None:
        pass

    def validate(self,
                 model: torch.nn.Module) -> None:

        model.eval()

        with torch.no_grad():
            for batch_idx, (data, targets) in tqdm(enumerate(self.loader)):
                data = data.to(self.device)
                targets = targets.to(self.device)
                scores = model(data)

                if self.batch_metrics:
                    for batch_metric in self.batch_metrics:
                        batch_metric.calculate(scores, targets)
                        batch_metric.notify(self.stage)

            if self.epoch_metrics:
                for epoch_metric in self.epoch_metrics:
                    epoch_metric.calculate(scores, targets)
                    epoch_metric.notify(self.stage)


class Runner(RunnerMediator):
    """This is a mediator that operates with its colleagues Trainer and
    Validator."""

    def __init__(self,
                 model: torch.nn.Module,
                 trainer: Trainer,
                 validator: Validator = None,
                 run_metrics: List[Metric] = None) -> None:
        super().__init__()
        self.model = model.to(self.device)
        self.trainer = trainer
        self.validator = validator
        self.run_metrics = run_metrics if isinstance(
            run_metrics, list) else [run_metrics]

    def train(self) -> None:
        try:
            self.trainer.train(self.model)
        except AttributeError as e:
            print(e)

    def validate(self) -> None:

        if self.validator:
            try:
                self.validator.validate(self.model)
            except AttributeError as e:
                print(e)

    def run(self, num_epochs=1) -> None:

        for _ in range(num_epochs):
            self.train()
            self.validate()

        with torch.no_grad():
            self.metrics_dict = {}

            if self.trainer:
                # for metric in self.run_metrics:
                #     self.metrics_dict[metric.name + "_TRAIN"] = 0

                total_size = len(self.trainer.loader.dataset)
                self.model.eval()
                all_scores = []
                all_targets = []

                for data, targets in self.trainer.loader:
                    current_batch_size = len(data)
                    all_scores.append(self.model(data))
                    all_targets.append(targets)

                for metric in self.run_metrics:
                    self.metrics_dict[metric.name + "_TRAIN"] = metric.calculate(
                        torch.cat(tensors=all_scores), torch.cat(tensors=all_targets))
                    metric.notify(self.metrics_dict)

            if self.validator:
                for metric in self.run_metrics:
                    self.metrics_dict[metric.name + "_VALID"] = 0

                total_size = len(self.validator.loader.dataset)
                self.model.eval()

                for data, targets in self.validator.loader:
                    current_batch_size = len(data)
                    scores = self.model(data)

                    for metric in self.run_metrics:
                        self.metrics_dict[metric.name + "_VALID"] += metric.calculate(
                            scores, targets) * current_batch_size / total_size

            # if self.trainer:
            #     train_metrics_values = {
            #         metric.name + self.trainer.stage metric.value
            #         for metric in self.trainer.epoch_metrics + self.trainer.batch_metrics
            #         if metric is not None}

            # if self.validator:
            #     valid_metrics_values = {
            #         metric.name + self.validator.stage: metric.value
            #         for metric in self.validator.epoch_metrics + self.validator.batch_metrics
            #         if metric is not None}
