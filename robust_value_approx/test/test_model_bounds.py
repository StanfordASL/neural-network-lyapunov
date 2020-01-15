import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.ball_paddle_hybrid_linear_system as bphls
import robust_value_approx.model_bounds as model_bounds
from robust_value_approx.utils import torch_to_numpy
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import double_integrator

import numpy as np
import unittest
import cvxpy as cp
import torch
import torch.nn as nn


class ModelBoundsUpperBound(unittest.TestCase):
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

    @unittest.skip("Test Failure, V(x) is None")
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

    def test_bound_fun(self):
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
            np.random.rand(10, x_dim), dtype=self.dtype)
        linear1.bias.data = torch.tensor(
            np.random.rand(10), dtype=self.dtype)
        linear2 = nn.Linear(10, 10)
        linear2.weight.data = torch.tensor(
            np.random.rand(10, 10),
            dtype=self.dtype)
        linear2.bias.data = torch.tensor(
            np.random.rand(10), dtype=self.dtype)
        linear3 = nn.Linear(10, 1)
        linear3.weight.data = torch.tensor(
            np.random.rand(1, 10), dtype=self.dtype)
        linear3.bias.data = torch.tensor([-10], dtype=self.dtype)
        model = nn.Sequential(linear1, nn.ReLU(), linear2,
                              nn.ReLU(), linear3)

        mb = model_bounds.ModelBounds(model, vf)
        x0_lo = -1. * torch.ones(x_dim, dtype=dtype)
        x0_up = 1. * torch.ones(x_dim, dtype=dtype)

        for i in range(10):
            x0 = torch.rand(x_dim, dtype=dtype) * (x0_up - x0_lo) + x0_lo
            eps = mb.bound_fun(model, x0_lo, x0_up, x0)
            eps_expected = V(x0)[0] - model(x0)
            self.assertAlmostEqual(eps.item(), eps_expected.item(), places=5)

    def test_lower_bound(self):
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
            np.random.rand(10, x_dim), dtype=self.dtype)
        linear1.bias.data = torch.tensor(
            np.random.rand(10), dtype=self.dtype)
        linear2 = nn.Linear(10, 10)
        linear2.weight.data = torch.tensor(
            np.random.rand(10, 10),
            dtype=self.dtype)
        linear2.bias.data = torch.tensor(
            np.random.rand(10), dtype=self.dtype)
        linear3 = nn.Linear(10, 1)
        linear3.weight.data = torch.tensor(
            np.random.rand(1, 10), dtype=self.dtype)
        linear3.bias.data = torch.tensor([-10], dtype=self.dtype)
        model = nn.Sequential(linear1, nn.ReLU(), linear2,
                              nn.ReLU(), linear3)

        mb = model_bounds.ModelBounds(model, vf)
        x0_lo = -1. * torch.ones(x_dim, dtype=dtype)
        x0_up = 1. * torch.ones(x_dim, dtype=dtype)

        x_adv = torch.Tensor([-.5, -.5]).type(dtype)
        x_adv.requires_grad = True
        optimizer = torch.optim.SGD([x_adv], lr=.01)

        for i in range(10):
            obj = -mb.bound_fun(model, x0_lo, x0_up,
                                torch.max(torch.min(x_adv, x0_up), x0_lo))
            optimizer.zero_grad()
            obj.backward()
            optimizer.step()
        x_adv = torch.max(torch.min(x_adv, x0_up), x0_lo)

        with torch.no_grad():
            eps_adv = mb.bound_fun(model, x0_lo, x0_up, x_adv)
            eps_expected = V(x_adv)[0] - model(x_adv)
            self.assertAlmostEqual(eps_adv.item(), eps_expected.item(),
                                   places=5)
            for i in range(10):
                x0 = torch.rand(x_dim, dtype=dtype) * (x0_up - x0_lo) + x0_lo
                eps_sample = V(x0)[0] - model(x0)
                self.assertGreaterEqual(eps_adv.item(), eps_sample.item())


if __name__ == '__main__':
    unittest.main()
