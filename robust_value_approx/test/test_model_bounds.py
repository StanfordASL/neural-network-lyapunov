import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.ball_paddle_hybrid_linear_system as bphls
import robust_value_approx.model_bounds as model_bounds
from robust_value_approx.utils import torch_to_numpy

import numpy as np
import unittest
import cvxpy as cp
import torch
import torch.nn as nn


class ModelBoundsUpperBound(unittest.TestCase):
    def setUp(self):
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

    def test_upper_bound(self):
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

        mb = model_bounds.ModelBounds(self.model, vf)
        x0_lo = torch.Tensor([0., 0., 0., 0., 0., 0.]).type(dtype)
        x0_up = torch.Tensor([0., 2., .1, 0., 0., 0.]).type(dtype)
        bound_opt = mb.upper_bound_opt(self.model, x0_lo, x0_up)
        Q1, Q2, q1, q2, k, G1, G2, h, A1, A2, b = torch_to_numpy(bound_opt)

        num_y = Q1.shape[0]
        num_gamma = Q2.shape[0]
        y = cp.Variable(num_y)
        gamma = cp.Variable(num_gamma, boolean=True)

        obj = cp.Minimize(.5 * cp.quad_form(y, Q1) + .5 *
                          cp.quad_form(gamma, Q2) + q1@y + q2@gamma + k)
        con = [
            A1@y + A2@gamma == b,
            G1@y + G2@gamma <= h,
        ]

        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        epsilon = obj.value

        V = vf.get_value_function()

        x0 = torch.Tensor(y.value[:sys.x_dim]).type(sys.dtype)
        value, _, _ = V(x0)
        z_nn = self.model(x0).item()

        self.assertTrue(np.abs((epsilon - (value - z_nn)) / epsilon) <= .01)

        for i in range(20):
            x0_sub = x0 + .1 * \
                torch.rand(sys.x_dim, dtype=sys.dtype) * (x0_up - x0_lo)
            x0_sub = torch.max(torch.min(x0_sub, x0_up), x0_lo)
            value_sub = None
            try:
                value_sub, _, _ = V(x0_sub)
            except AttributeError:
                # for some reason the solver didn't return anything
                pass
            if value_sub is not None:
                z_nn_sub = self.model(x0_sub).item()
                epsilon_sub = value_sub - z_nn_sub
                self.assertGreaterEqual(epsilon_sub, epsilon)


if __name__ == '__main__':
    unittest.main()
