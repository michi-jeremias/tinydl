from abc import ABC, abstractmethod
from typing import List

import torch
from tqdm import tqdm

# from tinydl.metric2 import Metric2
from tinydl.reporter2 import Reporter2
from tinydl.stage import Stage


class RunnerMediator2(ABC):

    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def train():
        """To be implemented by a Trainer()"""

    @abstractmethod
    def validate():
        """To be implemented by a Validator()"""


class Trainer2(RunnerMediator2):

    def __init__(self,
                 loader: torch.utils.data.DataLoader,
                 optimizer,
                 loss_fn,
                 batch_reporters: List[Reporter2] = [],
                 epoch_reporters: List[Reporter2] = [],
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
                        # batch_reporter.notify(self.stage)

        with torch.no_grad():
            if self.epoch_reporters:
                for epoch_reporter in self.epoch_reporters:
                    epoch_reporter.report(self.stage, scores, targets)
                    # epoch_metric.notify(self.stage)

    def validate() -> None:
        pass


class Validator2(RunnerMediator2):

    def __init__(self,
                 loader: torch.utils.data.DataLoader,
                 batch_reporters: List[Reporter2] = [],
                 epoch_reporters: List[Reporter2] = [],
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
                        # batch_reporter.notify(self.stage)

            if self.epoch_reporters:
                for epoch_reporter in self.epoch_reporters:
                    epoch_reporter.calculate(self.stage, scores, targets)
                    # epoch_metric.notify(self.stage)


class Runner2(RunnerMediator2):
    """This is a mediator that operates with its colleagues Trainer and
    Validator."""

    def __init__(self,
                 model: torch.nn.Module,
                 trainer: Trainer2,
                 validator: Validator2 = None,
                 run_reporters: List[Reporter2] = None) -> None:
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

            stage = Stage.VALID
            all_scores = []
            all_targets = []
            with torch.no_grad():
                self.model.eval()

                for data, targets in self.validator.loader:
                    all_scores.append(self.model(data))
                    all_targets.append(targets)

                for run_reporter in self.run_reporters:
                    run_reporter.report(stage, scores, targets)
                    run_reporter.report(stage, torch.cat(
                        tensors=all_scores), torch.cat(tensors=all_targets))