import unittest

from tinydl.hyperparameter import Hyperparameter


class TestHyperparameter(unittest.TestCase):

    def test_get_single_experiment(self):
        hparam = {"lr": [1e-3], "bs": [128]}
        hyper = Hyperparameter(hparam)

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
