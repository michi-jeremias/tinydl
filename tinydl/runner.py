from abc import ABC, abstractmethod
from typing import List

import torch
from tqdm import tqdm

from tinydl.reporter import Reporter
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
                 batch_reporters: List[Reporter] = [],
                 epoch_reporters: List[Reporter] = [],
                 ) -> None:
        super().__init__()
        self.loader = loader
        self.batch_reporters = batch_reporters if isinstance(
            batch_reporters, list) else [batch_reporters]
        self.epoch_reporters = epoch_reporters if isinstance(
            epoch_reporters, list) else [epoch_reporters]
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.stage = Stage.TRAIN

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
                if self.batch_reporters:
                    for batch_reporter in self.batch_reporters:
                        batch_reporter.report(self.stage, scores, targets)

        with torch.no_grad():
            if self.epoch_reporters:
                for epoch_reporter in self.epoch_reporters:
                    epoch_reporter.report(self.stage, scores, targets)

    def validate() -> None:
        pass


class Validator(RunnerMediator):

    def __init__(self,
                 loader: torch.utils.data.DataLoader,
                 batch_reporters: List[Reporter] = [],
                 epoch_reporters: List[Reporter] = [],
                 ) -> None:

        super().__init__()
        self.loader = loader
        self.batch_reporters = batch_reporters if isinstance(
            batch_reporters, list) else [batch_reporters]
        self.epoch_reporters = epoch_reporters if isinstance(
            epoch_reporters, list) else [epoch_reporters]
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

                if self.batch_reporters:
                    for batch_reporter in self.batch_reporters:
                        batch_reporter.report(self.stage, scores, targets)

            if self.epoch_reporters:
                for epoch_reporter in self.epoch_reporters:
                    epoch_reporter.report(self.stage, scores, targets)


class Runner(RunnerMediator):
    """This is a mediator that operates with its colleagues Trainer and
    Validator."""

    def __init__(self,
                 model: torch.nn.Module,
                 trainer: Trainer,
                 validator: Validator = None,
                 run_reporters: List[Reporter] = None) -> None:
        super().__init__()
        self.model = model.to(self.device)
        self.trainer = trainer
        self.validator = validator
        self.run_reporters = run_reporters if isinstance(
            run_reporters, list) else [run_reporters]

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

        if self.run_reporters and self.trainer:
            stage = Stage.TRAIN

            with torch.no_grad():
                self.model.eval()
                all_scores = []
                all_targets = []

                for data, targets in self.trainer.loader:
                    all_scores.append(self.model(data))
                    all_targets.append(targets)

                for run_reporter in self.run_reporters:
                    if run_reporter is not None:
                        run_reporter.report(stage=stage,
                                            scores=torch.cat(
                                                tensors=all_scores),
                                            targets=torch.cat(tensors=all_targets))

        if self.run_reporters and self.validator:
            stage = Stage.VALIDATION

            with torch.no_grad():
                self.model.eval()
                all_scores = []
                all_targets = []

                for data, targets in self.validator.loader:
                    all_scores.append(self.model(data))
                    all_targets.append(targets)

                for run_reporter in self.run_reporters:
                    if run_reporter is not None:
                        run_reporter.report(stage=stage,
                                            scores=torch.cat(
                                                tensors=all_scores),
                                            targets=torch.cat(tensors=all_targets))
