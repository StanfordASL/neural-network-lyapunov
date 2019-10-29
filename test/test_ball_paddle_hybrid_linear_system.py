from context import ball_paddle_hybrid_linear_system as bphls

import unittest
import numpy as np
import torch
import cvxpy as cp
from utils import torch_to_numpy


class BallPaddleHybridLinearSystemTest(unittest.TestCase):
    def setUp(self):
        """
        x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        u = [paddletheta_dot, paddlevy_dot]
        """
        self.dtype = torch.float64
        self.dt = .0025
        self.x_lo = torch.Tensor(
            [-1., 0., 0., -np.pi / 2,
             -1000., -1000., -1000.]).type(self.dtype)
        self.x_up = torch.Tensor(
            [1000., 2000., 1., np.pi / 2,
             10000., 10000., 10000.]).type(self.dtype)
        self.u_lo = torch.Tensor([-1000., -1000.]).type(self.dtype)
        self.u_up = torch.Tensor([1000., 1000.]).type(self.dtype)
        self.sys = bphls.get_ball_paddle_hybrid_linear_system(
            self.dtype, self.dt, self.x_lo, self.x_up, self.u_lo, self.u_up)

    def test_ball_paddle_sim(self):
        (Aeq_slack, Aeq_alpha,
         Ain_x, Ain_u, Ain_slack, Ain_alpha,
         rhs_in) = torch_to_numpy(
                self.sys.mixed_integer_constraints(
                        self.x_lo, self.x_up, self.u_lo, self.u_up))

        if len(Aeq_slack.shape) == 1:
            Aeq_slack = Aeq_slack.reshape((-1, 1))
        num_slack = Aeq_slack.shape[1]
        if len(Aeq_alpha.shape) == 1:
            Aeq_alpha = Aeq_alpha.reshape((-1, 1))
        num_alpha = Aeq_alpha.shape[1]

        slack = cp.Variable(num_slack)
        alpha = cp.Variable(num_alpha, boolean=True)

        x0 = np.array([0., 1., .5, 0., 0., 0., 5.])
        u0 = np.array([0., 0.])

        obj = cp.Minimize(0.)
        ball_traj = []
        paddle_traj = []
        for i in range(10):
            con = [
                Ain_x@x0 +
                Ain_u@u0 +
                Ain_slack@slack +
                Ain_alpha@alpha <= rhs_in,
                cp.sum(alpha) == 1.]
            prob = cp.Problem(obj, con)
            prob.solve(solver=cp.CPLEX, verbose=False)
            xn = Aeq_slack@slack.value + Aeq_alpha@alpha.value
            x0 = xn
            ball_traj.append(xn[1])
            paddle_traj.append(xn[2])

        self.assertTrue(slack.value is not None)
        self.assertTrue(alpha.value is not None)


if __name__ == '__main__':
    unittest.main()
