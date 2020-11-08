import neural_network_lyapunov.test.train_2d_lyapunov_utils as \
    train_2d_lyapunov_utils

import unittest

import os

import torch

import matplotlib.pyplot as plt
import numpy as np


class TestPlotSublevelSet(unittest.TestCase):
    def test(self):
        relu = torch.load(
            os.path.dirname(os.path.realpath(__file__)) +
            "/data/johansson_system1_lyapunov_no_bias.pt")["relu"]
        fig, ax = plt.subplots()
        x = train_2d_lyapunov_utils.plot_sublevel_set(
            ax, relu, 0., torch.tensor([0., 0.], dtype=torch.float64),
            R=torch.eye(2, dtype=torch.float64), upper_bound=0.1)
        relu_value = relu(x)
        np.testing.assert_array_almost_equal(
            relu_value.detach().numpy(), np.full(relu_value.shape, 0.1))


if __name__ == "__main__":
    unittest.main()
