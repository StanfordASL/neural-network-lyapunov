import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.model_bounds as model_bounds
from robust_value_approx.utils import torch_to_numpy
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import double_integrator

import numpy as np
import unittest
import cvxpy as cp
import torch
import torch.nn as nn


class ModelBoundsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        np.random.seed(123)

    def test_eps_opt(self):
        dt = 1.
        dtype = torch.float64
        (A_c, B_c) = double_integrator.double_integrator_dynamics(dtype)
        x_dim = A_c.shape[1]
        u_dim = B_c.shape[1]
        A = torch.eye(x_dim, dtype=dtype) + dt * A_c
        B = dt * B_c
        sys = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)
        c = torch.zeros(x_dim, dtype=dtype)
        P = torch.cat((-torch.eye(x_dim+u_dim),
                       torch.eye(x_dim+u_dim)), 0).type(dtype)
        x_lo = -10. * torch.ones(x_dim, dtype=dtype)
        x_up = 10. * torch.ones(x_dim, dtype=dtype)
        u_lo = -1. * torch.ones(u_dim, dtype=dtype)
        u_up = 1. * torch.ones(u_dim, dtype=dtype)
        q = torch.cat((-x_lo, -u_lo, x_up, u_up), 0).type(dtype)
        sys.add_mode(A, B, c, P, q)
        R = torch.eye(sys.u_dim)
        Q = torch.eye(sys.x_dim)
        N = 5
        vf = value_to_optimization.ValueFunction(sys, N,
                                                 x_lo, x_up, u_lo, u_up)
        vf.set_cost(Q=Q, R=R)
        vf.set_terminal_cost(Qt=Q, Rt=R)
        V = vf.get_value_function()

        linear1 = nn.Linear(x_dim, 10)
        linear1.weight.data = torch.tensor(
            np.random.rand(10, x_dim), dtype=dtype)
        linear1.bias.data = torch.tensor(
            np.random.rand(10), dtype=dtype)
        linear2 = nn.Linear(10, 10)
        linear2.weight.data = torch.tensor(
            np.random.rand(10, 10),
            dtype=dtype)
        linear2.bias.data = torch.tensor(
            np.random.rand(10), dtype=dtype)
        linear3 = nn.Linear(10, 1)
        linear3.weight.data = torch.tensor(
            np.random.rand(1, 10), dtype=dtype)
        linear3.bias.data = torch.tensor([-10], dtype=dtype)
        model = nn.Sequential(linear1, nn.ReLU(), linear2,
                              nn.ReLU(), linear3)

        mb = model_bounds.ModelBounds(vf, model)
        x0_lo = -1. * torch.ones(x_dim, dtype=dtype)
        x0_up = 1. * torch.ones(x_dim, dtype=dtype)

        eps_opt_coeffs = mb.epsilon_opt(model, x0_lo, x0_up)
        (Q0, Q1, Q2, q0, q1, q2, k,
         G0, G1, G2, h,
         A0, A1, A2, b) = torch_to_numpy(eps_opt_coeffs)

        num_y = Q1.shape[0]
        num_gamma = Q2.shape[0]
        x = cp.Variable(x_dim)
        y = cp.Variable(num_y)
        gamma = cp.Variable(num_gamma, boolean=True)

        obj = cp.Minimize(.5 * cp.quad_form(x, Q0) +
                          .5 * cp.quad_form(y, Q1) +
                          .5 * cp.quad_form(gamma, Q2) +
                          q0@x + q1@y + q2@gamma + k)
        con = [
            A0@x + A1@y + A2@gamma == b,
            G0@x + G1@y + G2@gamma <= h,
        ]

        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)

        epsilon = torch.Tensor([obj.value]).type(dtype)
        x_adv = torch.Tensor(x.value).type(dtype)
        epsilon_expected = V(x_adv)[0] - model(x_adv)
        self.assertAlmostEqual(epsilon.item(),
                               epsilon_expected.item(), places=5)
        self.assertTrue(torch.all(x_adv <= x0_up))
        self.assertTrue(torch.all(x_adv >= x0_lo))


if __name__ == '__main__':
    unittest.main()
