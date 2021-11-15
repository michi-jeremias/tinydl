from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from deeplearning.metric import Metric
from deeplearning.modelinit import init_normal


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


class Runner(RunnerMediator):
    """This is a mediator that operates with its colleagues Trainer and
    Validator."""

    def __init__(
            self,
            model,
            optimizer,
            loss_fn,
            trainer=None,
            validator=None) -> None:
        super().__init__()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
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

    def run(self) -> None:
        self.train()
        self.validate()


class Trainer(RunnerMediator):

    def __init__(self, loader, metrics) -> None:
        super().__init__()
        self.loader = loader
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    def train(self, model, optimizer, loss_fn) -> None:
        print(f"Trainer.train")
        model.train()

        for batch_idx, (data, targets) in tqdm(enumerate(self.loader)):
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Forward
            scores = model(data)
            self.loss = loss_fn(scores, targets)

            # Backward
            optimizer.zero_grad()
            self.loss.backward()
            optimizer.step()

        with torch.no_grad():
            for metric in self.metrics:
                metric.calculate(scores, targets)
                metric.notify()

    def validate() -> None:
        pass


class Validator(RunnerMediator):

    def __init__(self, loader, metrics) -> None:
        super().__init__()
        self.loader = loader
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    def train() -> None:
        pass

    def validate(self, model) -> None:
        print("Validator.validate")
        model.eval()

        with torch.no_grad():
            for batch_idx, (data, targets) in tqdm(enumerate(self.loader)):
                data = data.to(self.device)
                targets = targets.to(self.device)
                scores = model(data)

            for metric in self.metrics:
                metric.calculate(scores, targets)
                metric.notify()
