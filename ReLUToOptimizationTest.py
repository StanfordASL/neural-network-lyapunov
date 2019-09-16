import ReLUToOptimization
import unittest
import numpy as np
import torch
import torch.nn as nn


class TestReLUGivenActivationPath(unittest.TestCase):
    def setUp(self):
        self.datatype = torch.float
        self.linear1 = nn.Linear(2, 3)
        self.linear1.weight.data = torch.tensor(
            [[1, 2], [3, 4], [5, 6]], dtype=self.datatype)
        self.linear1.bias.data = torch.tensor([2, 3, 4], dtype=self.datatype)
        self.linear2 = nn.Linear(3, 4)
        self.linear2.weight.data = torch.tensor(
            [[-1, 0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1, 4, 6]],
            dtype=self.datatype)
        self.linear2.bias.data = torch.tensor(
            [4, -1, -2, 3], dtype=self.datatype)
        self.linear3 = nn.Linear(4, 1)
        self.linear3.weight.data = torch.tensor(
            [4, 5, 6, 7], dtype=self.datatype)
        self.linear3.bias.data = torch.tensor(-10, dtype=self.datatype)
        self.model = nn.Sequential(self.linear1, nn.ReLU(), self.linear2,
                                   nn.ReLU(),
                                   self.linear3)

    def test_compute_relu_activation_path(self):
        x = torch.tensor([-6, 4], dtype=self.datatype)
        activation_path = ReLUToOptimization.ComputeReLUActivationPath(
            self.model, x)
        self.assertEqual(len(activation_path), 2)
        self.assertEqual(len(activation_path[0]), 3)
        self.assertEqual(len(activation_path[1]), 4)
        x_linear1 = self.linear1.forward(x)
        x_relu1 = nn.ReLU().forward(x_linear1)
        for i in range(3):
            self.assertEqual(x_linear1[i] >= 0, activation_path[0][i])
        x_linear2 = self.linear2.forward(x_relu1)
        for i in range(4):
            self.assertEqual(x_linear2[i] >= 0, activation_path[1][i])

    def test_relu_given_activation_path(self):
        def test_relu_given_activation_path_util(self, x):
            activation_path = ReLUToOptimization.ComputeReLUActivationPath(
                self.model, x)
            (g, h, P, q) = ReLUToOptimization.ReLUGivenActivationPath(
                self.model, 2, activation_path)
            output_expected = self.model.forward(x)
            output = g.T @ x.reshape((2, 1)) + h
            self.assertAlmostEqual(output, output_expected, 10)
            self.assertTrue(torch.all(torch.le(P @ (x.reshape((-1, 1))), q)))

        test_relu_given_activation_path_util(
            self, torch.tensor([-6, 4], dtype=self.datatype))
        test_relu_given_activation_path_util(
            self, torch.tensor([-10, 4], dtype=self.datatype))
        test_relu_given_activation_path_util(
            self, torch.tensor([3, -4], dtype=self.datatype))
        test_relu_given_activation_path_util(
            self, torch.tensor([-3, -4], dtype=self.datatype))


if __name__ == "__main__":
    unittest.main()
