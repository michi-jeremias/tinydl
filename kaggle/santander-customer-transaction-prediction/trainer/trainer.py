from abc import ABC, abstractmethod
from trainer.const import LOGPATH
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
# writer = SummaryWriter(log_dir=f'{LOGPATH}/MNIST')

from model.utils import init_normal, init_xavier


class Trainer():

    def __init__(self, model, optimizer) -> None:
        self.epochs_trained = 0
        self.model = model
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.weight_decay = self.optimizer.param_groups[0]['weight_decay']

    def train(self, num_epochs) -> None:
        for epoch in range(num_epochs):
            print(
                f"[{self.epochs_trained + epoch + 1}/"
                f"{self.epochs_trained + num_epochs}]")
        self.epochs_trained += num_epochs

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
