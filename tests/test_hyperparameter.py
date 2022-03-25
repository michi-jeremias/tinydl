import unittest

from tinydl.hyperparameter import Hyperparameter


class TestHyperparameter(unittest.TestCase):

    def test_get_single_experiment(self):
        hparam = {"lr": [1e-3], "bs": [128]}
        hyper = Hyperparameter(hparam)
        experiments = hyper.get_experiments()
        experiment_list = []
        for experiment in experiments:
            experiment_list.append(experiment)

        expected_result = [
            {"lr": [1e-3], "bs": [128]}
        ]
        self.assertEqual(experiment_list, expected_result)

    def test_num_single_experiments(self):
        hparam = {"lr": [1e-3], "bs": [128]}
        hyper = Hyperparameter(hparam)
        self.assertEqual(hyper.num_experiments, 1)

    def test_num_single_experiments_no_list(self):
        hparam = {"lr": 1e-3, "bs": 128}
        hyper = Hyperparameter(hparam)
        self.assertEqual(hyper.num_experiments, 1)

    def test_num_two_experiments(self):
        hparam = {"lr": [1e-3, 1e-4], "bs": [128]}
        hyper = Hyperparameter(hparam)
        self.assertEqual(hyper.num_experiments, 2)

    def test_num_four_experiments(self):
        hparam = {"lr": [1e-3, 1e-4], "bs": [128, 256]}
        hyper = Hyperparameter(hparam)
        self.assertEqual(hyper.num_experiments, 4)

    def test_four_experiments(self):
        hparam = {"lr": [1e-3, 1e-4], "bs": [128, 256]}
        hyper = Hyperparameter(hparam)
        experiments = hyper.get_experiments()
        experiment_list = []
        for experiment in experiments:
            experiment_list.append(experiment)

        expected_result = [
            {"lr": 1e-3, "bs": 128},
            {"lr": 1e-4, "bs": 128},
            {"lr": 1e-3, "bs": 256},
            {"lr": 1e-4, "bs": 256},
        ]
        self.assertEqual(experiment_list, expected_result)
