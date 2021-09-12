from abc import ABC, abstractmethod
from trainer.const import LOGPATH
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
# writer = SummaryWriter(log_dir=f'{LOGPATH}/MNIST')

from model.utils import init_normal, init_xavier


class Trainer():

    def __init__(self, model, optimizer, loader, loss_fn, metrics_fn) -> None:
        """Interface for training models.

        Parameters
        ----------
        model : PyTorch neural network
        optimizer : torch.optim object
        loader : a container class for torch DataLoaders
        loss_fn : the loss function
        metrics_fn : function reporting a metric
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.epochs_trained = 0

        # Parameters
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loader = loader
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

        self.lr = self.optimizer.param_groups[0]['lr']
        self.weight_decay = self.optimizer.param_groups[0]['weight_decay']

    def __call__(self):
        print(f"Model: {self.model}")
        print(f"Loss function: {self.loss_fn}")
        print(f"Optimizer: {self.optimizer}")

    def train(self, num_epochs) -> None:
        """Trains self.model for num_epochs epochs.

        Parameters
        ----------
        num_epochs : int
        """

        for epoch in range(num_epochs):
            print(
                f"[{self.epochs_trained + epoch + 1}/"
                f"{self.epochs_trained + num_epochs}]")
            self.validate()

            for batch_idx, (data, targets) in enumerate(self.loader.train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward
                scores = self.model(data)
                loss = self.loss_fn(scores, targets)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx == 0:
                    print(loss)

        self.epochs_trained += num_epochs

    def validate(self) -> None:
        """Reports a metric from self.metrics_fn on the validation set."""
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():

            for x, y in self.loader.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                scores = self.model(x)
                predictions += scores.tolist()
                actuals += y.tolist()

        print(f"Validation ROC: {self.metrics_fn(actuals, predictions)}")
        self.model.train()

    def reset(self) -> None:
        self.model.apply(init_normal)
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.epochs_trained = 0

    # @abstractmethod
    # def out(self):
    #     pass

    # @abstractmethod
    # def save_model(self) -> None:
    #     pass

    # @abstractmethod
    # def load_model(self, file) -> None:
    #     pass


class TensorboardLogger:
    pass


class ScalarLogger(TensorboardLogger):
    pass


class ImageLogger(TensorboardLogger):
    pass
