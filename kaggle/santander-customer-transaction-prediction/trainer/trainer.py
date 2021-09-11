from abc import ABC, abstractmethod
from trainer.const import LOGPATH
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
# writer = SummaryWriter(log_dir=f'{LOGPATH}/MNIST')

from model.utils import init_normal, init_xavier


class Trainer():

    def __init__(self, model, optimizer, loader) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.epochs_trained = 0
        self.loader = loader
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.weight_decay = self.optimizer.param_groups[0]['weight_decay']

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
        self.epochs_trained += num_epochs

    def validate(self) -> None:
        """Reports a metric on the validation set."""
        print("Validation statistic: ")

        self.model.eval()
        saved_preds = []
        true_labels = []

        with torch.no_grad():
            for x, y in self.loader.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                scores = self.model(x)
                saved_preds += scores.tolist()
                true_labels += y.tolist()
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
