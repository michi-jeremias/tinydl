import itertools


class Hyperparameter():
    """This class takes in a dictionary and returns all permutations of
    values in this dictionary with a generator."""

    def __init__(self, hparam: dict) -> None:
        self.hparam = {}
        for key, value in hparam.items():
            self.hparam[key] = value if isinstance(value, list) else [value]
        self.experiments = self._generate_experiments(self.hparam)
        self.num_experiments = len(self.experiments)

    def _generate_experiments(self, hparam: dict) -> None:
        """Generate all permutations of hyperparameters from self.hparam"""
        
        keys, values = zip(*hparam.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return experiments

    def get_experiment(self) -> dict:
        """This generator returns an experiment. An experiment is a
        unique set of hyperparameters."""
        
        yield_index = 0
        
        while yield_index < self.num_experiments:
            experiment = self.experiments[yield_index]
            yield_index += 1
            yield experiment
