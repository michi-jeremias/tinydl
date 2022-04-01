"""Example implementation of AlexNet for MNIST.
Original paper: https://arxiv.org/abs/1404.5997v2.
Start tensorboard with 'tensorboard --logdir=runs' after all runs
are finished."""

from math import ceil
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.transforms import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.optim as optim

from tinydl.metric import BinaryCrossentropy, RocAuc
from tinydl.modelinit import init_xavier
from tinydl.metric import CrossEntropy
from tinydl.reporter import TensorboardScalarReporter, TensorboardHparamReporter
from tinydl.hyperparameter import Hyperparameter
from tinydl.runner import Runner, Trainer, Validator


# Dataset
transforms = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(1)], [0.5 for _ in range(1)]
        ),
    ]
)
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms,
                         download=True)
train_ds, validation_ds = random_split(
    dataset, [int(0.8 * len(dataset)), ceil(0.2 * len(dataset))])


# Loss function
loss_fn = nn.CrossEntropyLoss()


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
        dataset=validation_ds,
        batch_size=experiment["batchsize"])

    model = models.alexnet(pretrained=False)
    model.classifier.add_module("7", nn.Linear(
        in_features=1000, out_features=10, bias=False))

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=experiment["lr"],
        weight_decay=1e-4
    )

    tbscalar_reporter = TensorboardScalarReporter(hparam=experiment)
    tbscalar_reporter.add_metrics([CrossEntropy()])

    tbscalar_reporter_val = TensorboardScalarReporter(hparam=experiment)
    tbscalar_reporter_val.add_metrics([CrossEntropy()])

    tbhparam_reporter = TensorboardHparamReporter(hparam=experiment)
    tbhparam_reporter.add_metrics([CrossEntropy()])

    TRAINER = Trainer(
        loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        batch_reporters=tbscalar_reporter
    )

    VALIDATOR = Validator(
        loader=val_loader,
        batch_reporters=tbscalar_reporter_val
    )

    RUNNER = Runner(
        model=model,
        trainer=TRAINER,
        validator=VALIDATOR,
        run_reporters=tbhparam_reporter
    )

    RUNNER.run(3)
