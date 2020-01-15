import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.ball_paddle_hybrid_linear_system as bphls
from robust_value_approx.utils import train_model

import numpy as np
import unittest
import torch
import torch.nn as nn


class ModelTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.dtype = torch.float64
        self.linear1 = nn.Linear(6, 10)
        self.linear1.weight.data = torch.tensor(
            np.random.rand(10, 6), dtype=self.dtype)
        self.linear1.bias.data = torch.tensor(
            np.random.rand(10), dtype=self.dtype)
        self.linear2 = nn.Linear(10, 10)
        self.linear2.weight.data = torch.tensor(
            np.random.rand(10, 10),
            dtype=self.dtype)
        self.linear2.bias.data = torch.tensor(
            np.random.rand(10), dtype=self.dtype)
        self.linear3 = nn.Linear(10, 1)
        self.linear3.weight.data = torch.tensor(
            np.random.rand(1, 10), dtype=self.dtype)
        self.linear3.bias.data = torch.tensor([-10], dtype=self.dtype)
        self.model = nn.Sequential(self.linear1, nn.ReLU(), self.linear2,
                                   nn.ReLU(),
                                   self.linear3)

    @unittest.skip("Test Failure when GUROBI has issues generating samples")
    def test_model_training(self):
        dtype = torch.float64
        dt = .01
        N = 10
        x_lo = torch.Tensor(
            [-1., -1., 0., -1e6, -1e6, -1e6]).type(dtype)
        x_up = torch.Tensor(
            [1., 10., 2., 1e6, 1e6, 1e6]).type(dtype)
        u_lo = torch.Tensor([-np.pi / 2, -1e7]).type(dtype)
        u_up = torch.Tensor([np.pi / 2, 1e7]).type(dtype)
        sys = bphls.get_ball_paddle_hybrid_linear_system(
            dtype, dt, x_lo, x_up, u_lo, u_up)
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)

        Q = torch.ones(sys.x_dim, sys.x_dim) * 0.1
        q = torch.ones(sys.x_dim) * 0.1
        R = torch.ones(sys.u_dim, sys.u_dim) * 2.
        r = torch.ones(sys.u_dim) * 0.1
        vf.set_cost(Q=Q, R=R, q=q, r=r)
        vf.set_terminal_cost(Qt=Q, Rt=R, qt=q, rt=r)
        xN = torch.Tensor([np.nan, .5, 0., np.nan, 0., np.nan])
        vf.set_constraints(xN=xN)

        x0_lo = torch.Tensor([0., 0., 0., 0., 0., 0.]).type(dtype)
        x0_up = torch.Tensor([0., 2., .1, 0., 0., 0.]).type(dtype)
        num_breaks = [1, 3, 3, 1, 3, 1]
        x_samples, v_samples = vf.get_value_sample_grid(
            x0_lo, x0_up, num_breaks)

        # check loss pre training
        res = self.model(x_samples) - v_samples
        loss_before = (res.t() @ res).item()

        train_model(self.model, x_samples, v_samples,
                    num_epoch=1000, batch_size=10)

        res = self.model(x_samples) - v_samples
        loss_after = (res.t() @ res).item()

        self.assertLess(loss_after, loss_before)


if __name__ == '__main__':
    unittest.main()
