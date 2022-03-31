"""Example implementation of AlexNet for MNIST. Original paper: https://arxiv.org/abs/1404.5997v2"""


import torch
import torch.nn as nn
from torchvision.datasets import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from tinydl.metric import BinaryCrossentropy, RocAuc
from tinydl.modelinit import init_xavier
from tinydl.metric import BinaryCrossentropy
from tinydl.reporter import TensorboardScalarReporter, TensorboardHparamReporter
from tinydl.stage import Stage
from tinydl.hyperparameter import Hyperparameter


# Dataset
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms,
                         download=True)

# Loss function
loss_fn = nn.BCELoss()


# Hyperparameters
hparam = {
    "batchsize": [128, 256, 512, 1024],
    "lr": [2e-3, 2e-4]
}

hyperparameter = Hyperparameter(hparam)


# Run
for experiment in hyperparameter.get_experiments():
    print(f"NEW RUN: {experiment}")

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=experiment["batchsize"],
        shuffle=True)
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=experiment["batchsize"])
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=experiment["batchsize"])

    model = DeepConvNet(input_size=400, hidden_dim=100)
    init_xavier(model)
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=experiment["lr"],
        weight_decay=1e-4
    )

    # console_reporter = ConsoleReporter()
    # console_reporter.add_metrics([BinaryCrossentropy()])

    tbscalar_reporter = TensorboardScalarReporter(hparam=experiment)
    tbscalar_reporter.add_metrics([BinaryCrossentropy(), RocAuc()])

    # console_reporter_val = ConsoleReporter()
    # console_reporter_val.add_metrics(BinaryCrossentropy())

    tbscalar_reporter_val = TensorboardScalarReporter(hparam=experiment)
    tbscalar_reporter_val.add_metrics([BinaryCrossentropy(), RocAuc()])

    tbhparam_reporter = TensorboardHparamReporter(hparam=experiment)
    tbhparam_reporter.add_metrics([BinaryCrossentropy(), RocAuc()])

    TRAINER = Trainer(
        loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        batch_reporters=tbscalar_reporter,
        # epoch_reporters=console_reporter
    )

    VALIDATOR = Validator(
        loader=val_loader,
        batch_reporters=tbscalar_reporter_val,
    )

    RUNNER = Runner(
        model=model,
        trainer=TRAINER,
        validator=VALIDATOR,
        run_reporters=tbhparam_reporter
    )

    RUNNER.run(3)


# comment mnist above and uncomment below if train on CelebA
#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
