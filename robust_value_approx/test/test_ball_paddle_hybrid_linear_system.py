import robust_value_approx.ball_paddle_hybrid_linear_system as bphls

import unittest
import numpy as np
import torch
import cvxpy as cp
from robust_value_approx.utils import torch_to_numpy
# import matplotlib.pyplot as plt


class BallPaddleHybridLinearSystemTest(unittest.TestCase):
    def setUp(self):
        """
        x = [ballx, bally, paddley, ballvx, ballvy, paddlevy]
        u = [paddletheta_dot, paddlevy_dot]
        """
        self.dtype = torch.float64
        self.dt = .0025
        self.x_lo = torch.Tensor(
            [-1., 0., 0.,
             -1000., -1000., -1000.]).type(self.dtype)
        self.x_up = torch.Tensor(
            [1000., 2000., 1.,
             10000., 10000., 10000.]).type(self.dtype)
        self.u_lo = torch.Tensor([-np.pi / 2, -1000.]).type(self.dtype)
        self.u_up = torch.Tensor([np.pi / 2, 1000.]).type(self.dtype)
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

        x0 = np.array([0., 1., .5, 0., 0., 5.])
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
            prob.solve(solver=cp.GUROBI, verbose=False)
            xn = Aeq_slack@slack.value + Aeq_alpha@alpha.value
            x0 = xn
            ball_traj.append(xn[1])
            paddle_traj.append(xn[2])

        self.assertTrue(slack.value is not None)
        self.assertTrue(alpha.value is not None)


class BallPaddleHybridLinearSystemVelCtrlTest(unittest.TestCase):
    def setUp(self):
        """
        x = [ballx, bally, paddley, paddletheta, ballvx, ballvy, paddlevy]
        u = [paddletheta_dot, paddlevy_dot]
        """
        self.dtype = torch.float64
        self.dt = .05
        self.x_lo = torch.Tensor(
            [-10., -10., -10., -2.*np.pi,
             -1000., -1000., -1000.]).type(self.dtype)
        self.x_up = torch.Tensor(
            [10., 10., 10., 2.*np.pi,
             1000., 1000., 1000.]).type(self.dtype)
        self.u_lo = torch.Tensor([-100, -100.]).type(self.dtype)
        self.u_up = torch.Tensor([100, 100.]).type(self.dtype)
        self.sys = bphls.get_ball_paddle_hybrid_linear_system_vel_ctrl(
            self.dtype, self.dt, self.x_lo, self.x_up, self.u_lo, self.u_up,
            paddle_angles=torch.Tensor([-.5, 0., .5]).type(self.dtype),
            midpoint=True)

    def test_ball_paddle_sim(self):
        (Aeq_slack, Aeq_alpha,
         Ain_x, Ain_u, Ain_slack, Ain_alpha,
         rhs_in) = torch_to_numpy(self.sys.mixed_integer_constraints(),
                                  squeeze=False)
        slack = cp.Variable(Ain_slack.shape[1])
        alpha = cp.Variable(Ain_alpha.shape[1], boolean=True)
        x0 = np.array([0., 1., .5, .5, 0., 0., 0.])
        u0 = np.array([0., 0.])

        obj = cp.Minimize(0.)
        ball_traj_x = []
        ball_traj_y = []
        paddle_traj = []
        for i in range(30):
            con = [
                Ain_x@x0 +
                Ain_u@u0 +
                Ain_slack@slack +
                Ain_alpha@alpha <= rhs_in,
                cp.sum(alpha) == 1.]
            prob = cp.Problem(obj, con)
            prob.solve(solver=cp.GUROBI, verbose=False)
            xn = Aeq_slack@slack.value + Aeq_alpha@alpha.value
            x0 = xn
            ball_traj_x.append(xn[0])
            ball_traj_y.append(xn[1])
            paddle_traj.append(xn[2])
        # plt.plot(ball_traj_x)
        # plt.plot(ball_traj_y)
        # plt.plot(paddle_traj)
        # plt.show()
        self.assertTrue(slack.value is not None)
        self.assertTrue(alpha.value is not None)
        self.assertGreater(ball_traj_y[0], max(ball_traj_y[1:]))
        self.assertGreater(ball_traj_x[-1], ball_traj_x[0])


if __name__ == '__main__':
    unittest.main()
