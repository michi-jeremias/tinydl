from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from tinydl.hyperparameter import Hyperparameter
from tinydl.metric import Metric
from tinydl.modelinit import init_normal


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

    def __init__(self, loader, batch_metrics, epoch_metrics, optimizer, loss_fn) -> None:
        super().__init__()
        self.loader = loader
        self.batch_metrics = batch_metrics if isinstance(
            batch_metrics, list) else [batch_metrics]
        self.epoch_metrics = epoch_metrics if isinstance(
            epoch_metrics, list) else [epoch_metrics]
        self.optimizer = optimizer
        self.loss_fn = loss_fn

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
                for metric in self.batch_metrics:
                    metric.calculate(scores, targets)
                    metric.notify()

        with torch.no_grad():
            for metric in self.epoch_metrics:
                metric.calculate(scores, targets)
                metric.notify()

    def validate() -> None:
        pass


class Validator(RunnerMediator):

    def __init__(self,
                 loader: torch.utils.data.DataLoader,
                 batch_metrics: list(Metric),
                 epoch_metrics: list(Metric)) -> None:
        super().__init__()
        self.loader = loader
        self.batch_metrics = batch_metrics if isinstance(
            batch_metrics, list) else [batch_metrics]
        self.epoch_metrics = epoch_metrics if isinstance(
            epoch_metrics, list) else [epoch_metrics]

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

                for metric in self.batch_metrics:
                    metric.calculate(scores, targets)
                    metric.notify()

            for metric in self.epoch_metrics:
                metric.calculate(scores, targets)
                metric.notify()


class Runner(RunnerMediator):
    """This is a mediator that operates with its colleagues Trainer and
    Validator."""

    def __init__(self,
                 model: torch.nn.Module,
                 trainer: Trainer = None,
                 validator: Validator = None) -> None:
        super().__init__()
        self.model = model.to(self.device)
        self.trainer = trainer
        self.validator = validator

    def train(self) -> None:
        try:
            self.trainer.train(self.model, self.optimizer, self.loss_fn)
        except AttributeError:
            print("No trainer in runner.")

    def validate(self) -> None:
        try:
            self.validator.validate(self.model)
        except AttributeError:
            print("No validator in runner.")

    def run(self, num_epochs=1) -> None:
        for _ in range(num_epochs):
            self.train()
            self.validate()
