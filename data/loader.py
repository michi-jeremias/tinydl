class Loader():
    """This is a container for three data loaders."""

    def __init__(self, train_loader, val_loader=None, test_loader=None):
        self.train = train_loader
        self.val = val_loader
        self.test = test_loader

    @property
    def train_loader(self):
        return self.train

    @property
    def val_loader(self):
        if self.val is None:
            print("No validaton loader in container.")
        return self.val

    @property
    def test_loader(self):
        if self.test is None:
            print("No test loader in container.")
        return self.test
