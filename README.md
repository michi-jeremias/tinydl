# tinydl

tinydl (tiny deeplearning) is a Python library which aims to facilitate training and validation of deep learning models implemented with PyTorch. tinydl is following semantic versioning, see https://semver.org/

# Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tinydl.

```bash
pip install tinydl
```

# Usage

I have included a showcase in `example_alexnet_mnist.py`, feel free to check it out! All examples described in this README here are directly taken from this file. You can find it in the root directory of the repository.

## Hyperparameter

The Hyperparameter class is used to generate permutations across all values in the lists provided by an input dictionary. Hyperparameter.get_experiments() returns a `dict` for each permutation that you can then use in the training/validation algorithm. The number and names of hyperparameters are completely arbitrary which allows a great deal of flexibility when doing a hyperparameter search.

```Python
from tinydl.Hyperparameter import Hyperparameter
hparam = {
    "batchsize": [128, 256],
    "lr": [2e-3, 2e-4]
}
hyperparameter = Hyperparameter(hparam)
for experiment in hyperparameter.get_experiments():
    print(experiment)
```

This will generate the following output:

```Python
{'batchsize': 128, 'lr': 0.002}
{'batchsize': 128, 'lr': 0.0002}
{'batchsize': 256, 'lr': 0.002}
{'batchsize': 256, 'lr': 0.0002}
```

## Trainer, Validator, Runner

### Trainer

The Trainer class contains the training algorithm (forward, backward with `model.train()`), and within various hooks where the calculation and reporting of metrics will be executed. The Trainer class can be used on its own, or a Trainer can be added to a Runner class, which will then execute Trainer.train().

```Python
from tinydl.runner import Trainer
TRAINER = Trainer(
    loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    batch_reporters=tbscalar_reporter
)
TRAINER.train(model)
```

### Validator

The Validator class is very similar to the Trainer class. The difference is that there is no backward pass, the model runs in `model.eval()` and the values are calculated within a `torch.no_grad()` context manager.

```Python
from tinydl.runner import Validator
VALIDATOR = Validator(
    loader=val_loader,
    batch_reporters=tbscalar_reporter_val
)
VALIDATOR.validate(model)
```

### Runner

The `Runner` class takes in a `Trainer` and a `Validator`, and calls `Trainer.train()` and `Validator.validate()` respectively. You can specify the number of epochs that you want to train/validate by providing an argument `num_epochs: int` to `Runner.run()`. The Runner also takes in a list `run_reporters` which can be used to calculate and report metrics after all epochs have been trained/validated, i.e. at the end of a run. The use case is to report the final values of metrics after everything else is finished, e.g. to torch.utils.tensorboard.SummaryWriter.add_hparams() where you typically only want a single value for each metric, for each set of hyperparameters. This can be achieved by adding `tinydl.reporter.TensorboardHparamReporter` to run_reporters.

```Python
RUNNER = Runner(
    model=model,
    trainer=TRAINER,
    validator=VALIDATOR,
    run_reporters=tbhparam_reporter
)
RUNNER.run(3)
```

## Metric

The `Metric` class is used to calculate metrics (surprise!), and is usually not used directly, but rather added to a `Reporter`. The Reporter calls `Metric.calculate()` to generate a `Report` (see below). Metric.calculate() takes in two arguments: scores and targets. In an observer pattern Metric is a dependent (Observer).

## Reporter

The `Reporter` class generates and reports output by creating a `Report` based on the metrics that are added to a Reporter.
A Reporter is then assigned to a Trainer, a Validator or a Runner, and can be hooked into various places. Currently you can let a Reporter output a Report after a batch, after an epoch and after a run. In an observer pattern Reporter is a Subject (Observable).

```Python
from tinydl.metric import CrossEntropy
from tinydl.reporter import TensorboardScalarReporter, TensorboardHparamReporter

tbscalar_reporter = TensorboardScalarReporter(hparam=experiment)
tbscalar_reporter.add_metrics([CrossEntropy()])

tbhparam_reporter = TensorboardHparamReporter(hparam=experiment)
tbhparam_reporter.add_metrics([CrossEntropy()])
```

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate. I will try to get to it as quickly as possible.

# License

[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)
