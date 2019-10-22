from context import value_to_optimization
from context import ball_paddle_system
from context import model_bounds

import unittest
import cvxpy as cp
import torch
import torch.nn as nn
from utils import torch_to_numpy


class ModelBoundsUpperBound(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.linear1 = nn.Linear(3, 3)
        self.linear1.weight.data = torch.tensor(
            [[1, 2, 2.5], [3, 4, 4.5], [5, 6, 6.5]], dtype=self.dtype)
        self.linear1.bias.data = torch.tensor(
            [-11, 13, 4], dtype=self.dtype)
        self.linear2 = nn.Linear(3, 4)
        self.linear2.weight.data = torch.tensor(
            [[-1, 0.5, 1.5], [2, 5, 6], [-2, -3, -4], [1, 4, 6]],
            dtype=self.dtype)
        self.linear2.bias.data = torch.tensor(
            [4, -1, -2, 3], dtype=self.dtype)
        self.linear3 = nn.Linear(4, 1)
        self.linear3.weight.data = torch.tensor(
            [[4, 5, 6, 7]], dtype=self.dtype)
        self.linear3.bias.data = torch.tensor([-10], dtype=self.dtype)
        self.model = nn.Sequential(self.linear1, nn.ReLU(), self.linear2,
                                   nn.ReLU(),
                                   self.linear3)

    def test_upper_bound(self):
        sys = ball_paddle_system.BallPaddleSystem(dt=.01)

        vf = value_to_optimization.ValueFunction(sys, 10)
        Q = torch.ones(3, 3)*0.1
        q = torch.ones(3)*0.1
        R = torch.ones(1, 1)*2.
        r = torch.ones(1)*0.1
        vf.set_cost(Q=Q, R=R, q=q, r=r)
        vf.set_terminal_cost(Qt=Q, Rt=R, qt=q, rt=r)
        xN = torch.Tensor([0., .075, 0.]).type(sys.dtype)
        vf.set_constraints(xN=xN)

        mb = model_bounds.ModelBounds(self.model, vf)
        x_lo = torch.Tensor([0., 0., -10.]).type(sys.dtype)
        x_up = torch.Tensor([1., 1.5, 10.]).type(sys.dtype)
        bound_opt = mb.upper_bound_opt(self.model, x_lo, x_up)
        Q1, Q2, q1, q2, k, G1, G2, h, A1, A2, b = torch_to_numpy(bound_opt)

        num_y = Q1.shape[0]
        num_gamma = Q2.shape[0]
        y = cp.Variable(num_y)
        gamma = cp.Variable(num_gamma, boolean=True)

        obj = cp.Minimize(.5*cp.quad_form(y, Q1) + .5 *
                          cp.quad_form(gamma, Q2) + q1@y + q2@gamma + k)
        con = [
            A1@y + A2@gamma == b,
            G1@y + G2@gamma <= h,
        ]

        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.GUROBI, verbose=False)
        epsilon = obj.value

        V = vf.get_value_function()

        x0 = torch.Tensor(y.value[:3]).type(sys.dtype)
        value, _, _ = V(x0)
        z_nn = self.model(x0).item()

        self.assertAlmostEqual(epsilon, value - z_nn, places=4)

        for i in range(200):
            if i < 50:
                # uniform over the input space
                x0_sub = torch.rand(3, dtype=sys.dtype)*(x_up - x_lo) + x_lo
            else:
                # locally around the adversarial example
                x0_sub = x0 + .2*torch.rand(3, dtype=sys.dtype)*(x_up - x_lo)
                x0_sub = torch.max(torch.min(x0_sub, x_up), x_lo)
            value_sub = None
            try:
                value_sub, _, _ = V(x0_sub)
            except:
                pass
            if value_sub is not None:
                z_nn_sub = self.model(x0_sub).item()
                epsilon_sub = value_sub - z_nn_sub
                self.assertGreaterEqual(epsilon_sub, epsilon)


if __name__ == '__main__':
    unittest.main()
