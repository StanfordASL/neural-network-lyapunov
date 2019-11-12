import gurobipy
import numpy as np
import unittest
import torch
import torch.nn as nn

import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.relu_to_optimization as relu_to_optimization
import robust_value_approx.lyapunov as lyapunov

class TestLyapunovDiscreteTimeHybridSystem(unittest.TestCase):

    def setUp(self):
        """
        The piecewise affine system is from "Analysis of discrete-time
        piecewise affine and hybrid systems
        """
        self.dtype=torch.float64
        self.system1 = hybrid_linear_system.AutonomousHybridLinearSystem(
            2, self.dtype)
        self.system1.add_mode(
            torch.tensor([[-0.999, 0], [-0.139, 0.341]], dtype=self.dtype),
            torch.zeros((2,),dtype=self.dtype),
            torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=self.dtype),
            torch.tensor([1, 0, 0, 1], dtype=self.dtype))
        self.system1.add_mode(
            torch.tensor([[0.436, 0.323], [0.388, -0.049]], dtype=self.dtype),
            torch.zeros((2,),dtype=self.dtype),
            torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=self.dtype),
            torch.tensor([1, 0, 1, 0], dtype=self.dtype))
        self.system1.add_mode(
            torch.tensor([[-0.457, 0.215], [0.491, 0.49]], dtype=self.dtype),
            torch.zeros((2,), dtype=self.dtype),
            torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=self.dtype),
            torch.tensor([0, 1, 0, 1], dtype=self.dtype))
        self.system1.add_mode(
            torch.tensor([[-0.022, 0.344], [0.458, 0.271]], dtype=self.dtype),
            torch.zeros((2,), dtype=self.dtype),
            torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=self.dtype),
            torch.tensor([0, 1, 1, 0], dtype=self.dtype))

    def test_lyapunov_as_milp(self):
        dut = lyapunov.LyapunovDiscreteTimeHybridSystem(self.system1)
        # Construct a simple ReLU model with 2 hidden layers
        linear1 = nn.Linear(2, 3)
        linear1.weight.data = torch.tensor([[1, 2], [3, 4], [5, 6]],
                                           dtype=self.dtype)
        linear1.bias.data = torch.tensor([-11, 10, 5], dtype=self.dtype)
        linear2 = nn.Linear(3, 4)
        linear2.weight.data = torch.tensor(
                [[-1, -0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1.5, 4, 6]],
                dtype=self.dtype)
        linear2.bias.data = torch.tensor([-3, 2, 0.7, 1.5], dtype=self.dtype)
        linear3 = nn.Linear(4, 1)
        linear3.weight.data = torch.tensor([[4, 5, 6, 7]], dtype=self.dtype)
        linear3.bias.data = torch.tensor([-9], dtype=self.dtype)
        relu1 = nn.Sequential(
            linear1, nn.ReLU(), linear2, nn.ReLU(), linear3, nn.ReLU())

        dut.lyapunov_as_milp(relu1)
        

if __name__ == "__main__":
    unittest.main()
