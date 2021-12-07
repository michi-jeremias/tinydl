# tinydl

tinydl (tiny deeplearning) is a Python library which facilitates training and validation of deep learning models implemented with PyTorch.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tinydl.

```bash
pip install tinydl
```

## Usage

```python
import tinydl

# Instantiate Trainer
trainer = tinydl.Trainer(
    loader=train_loader   # torch.utils.data.DataLoader
)

# Instantiate Runner
runner = tinydl.Runner(
    model=model,          # torch.nn.Module
    optimizer=optimizer   # torch.optim
)

# returns 'phenomenon'
runner.run(5)             # run 5 epochs
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)