from abc import ABC, abstractmethod
from trainer.const import LOGPATH
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(log_dir=f'{LOGPATH}/MNIST')


class Trainer(ABC):

    def __init__(self, model, optimizer, loss_fn, train_loader,
                 val_loader, test_loader) -> None:
        pass

    @abstractmethod
    def train(self, num_epochs) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def out(self):
        pass

    @abstractmethod
    def save_model(self) -> None:
        pass

    @abstractmethod
    def load_model(self, file) -> None:
        pass


class TensorboardLogger:
    pass
