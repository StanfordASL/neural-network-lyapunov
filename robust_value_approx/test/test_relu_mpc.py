import robust_value_approx.relu_mpc as relu_mpc
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.value_to_optimization as value_to_optimization
import double_integrator

import os

import torch
import unittest


class TestReLUMPC(unittest.TestCase):
    def setUp(self):
        dtype = torch.float64
        self.dtype = dtype
        (A_c, B_c) = double_integrator.double_integrator_dynamics(dtype)
        x_dim = A_c.shape[1]
        u_dim = B_c.shape[1]
        # continuous to discrete using forward euler
        dt = 1.
        A = torch.eye(x_dim, dtype=dtype) + dt * A_c
        B = dt * B_c
        c = torch.zeros(x_dim, dtype=dtype)
        self.x_lo = -2. * torch.ones(x_dim, dtype=dtype)
        self.x_up = 2. * torch.ones(x_dim, dtype=dtype)
        self.u_lo = -1. * torch.ones(u_dim, dtype=dtype)
        self.u_up = 1. * torch.ones(u_dim, dtype=dtype)
        P = torch.cat((-torch.eye(x_dim+u_dim),
                       torch.eye(x_dim+u_dim)), 0).type(dtype)
        q = torch.cat((-self.x_lo, -self.u_lo,
                       self.x_up, self.u_up), 0).type(dtype)
        self.double_int = hybrid_linear_system.HybridLinearSystem(x_dim,
                                                                  u_dim,
                                                                  dtype)
        self.double_int.add_mode(A, B, c, P, q)
        # value function
        N = 6
        vf = value_to_optimization.ValueFunction(self.double_int, N,
                                                 self.x_lo, self.x_up,
                                                 self.u_lo, self.u_up)
        R = torch.eye(self.double_int.u_dim)
        vf.set_cost(R=R)
        vf.set_terminal_cost(Rt=R)
        xN = torch.Tensor([0., 0.]).type(dtype)
        vf.set_constraints(xN=xN)
        self.vf = vf
        self.vf_value_fun = vf.get_value_function()
        # should be trained from vf (minus one time step)
        self.model = torch.load(
            os.path.dirname(os.path.realpath(__file__)) +
            "/data/double_integrator_model.pt")

    def test_random_shooting(self):
        num_samples = 100
        ctrl = relu_mpc.RandomShootingMPC(self.vf, self.model, num_samples)
        for i in range(25):
            x0 = torch.rand(self.vf.sys.x_dim, dtype=self.dtype) *\
                (self.x_up - self.x_lo) + self.x_lo
            u_opt = self.vf_value_fun(x0)[1]
            if isinstance(u_opt, type(None)):
                continue
            u = ctrl.get_ctrl(x0)
            if ~isinstance(u_opt, type(None)):
                self.assertLessEqual(abs(u.item() - u_opt[0]), .05)

    def test_relu_mpc(self):
        ctrl = relu_mpc.ReLUMPC(self.vf, self.model, self.x_lo, self.x_up)
        for i in range(10):
            x0 = torch.rand(self.vf.sys.x_dim, dtype=self.dtype) *\
                (self.x_up - self.x_lo) + self.x_lo
            u_opt = self.vf_value_fun(x0)[1]
            if isinstance(u_opt, type(None)):
                continue
            u, x = ctrl.get_ctrl(x0)
            if ~isinstance(u_opt, type(None)):
                self.assertLessEqual(abs(u.item() - u_opt[0]), .05)


class TestQReLUMPC(unittest.TestCase):
    def setUp(self):
        dtype = torch.float64
        self.dtype = dtype
        (A_c, B_c) = double_integrator.double_integrator_dynamics(dtype)
        x_dim = A_c.shape[1]
        u_dim = B_c.shape[1]
        # continuous to discrete using forward euler
        dt = 1.
        A = torch.eye(x_dim, dtype=dtype) + dt * A_c
        B = dt * B_c
        sys = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)
        c = torch.zeros(x_dim, dtype=dtype)
        x_lo = -10. * torch.ones(x_dim, dtype=dtype)
        x_up = 10. * torch.ones(x_dim, dtype=dtype)
        u_lo = -1. * torch.ones(u_dim, dtype=dtype)
        u_up = 1. * torch.ones(u_dim, dtype=dtype)
        P = torch.cat((-torch.eye(x_dim+u_dim),
                       torch.eye(x_dim+u_dim)), 0).type(dtype)
        q = torch.cat((-x_lo, -u_lo, x_up, u_up), 0).type(dtype)
        sys.add_mode(A, B, c, P, q)
        # value function
        N = 5
        vf = value_to_optimization.ValueFunction(
            sys, N, x_lo, x_up, u_lo, u_up)
        Q = torch.eye(sys.x_dim)
        R = torch.eye(sys.u_dim)
        vf.set_cost(Q=Q, R=R)
        vf.set_terminal_cost(Qt=Q, Rt=R)
        # vf.set_constant_control(0)
        self.vf = vf
        self.vf_value_fun = vf.get_value_function()
        self.model = torch.load(
           os.path.dirname(os.path.realpath(__file__)) +
           "/data/double_integrator_q_model.pt")

    def test_qrelu_mpc(self):
        x0_lo = -1. * torch.ones(self.vf.sys.x_dim, dtype=self.dtype)
        x0_up = 1. * torch.ones(self.vf.sys.x_dim, dtype=self.dtype)
        u0_lo = -1. * torch.ones(self.vf.sys.u_dim, dtype=self.dtype)
        u0_up = 1. * torch.ones(self.vf.sys.u_dim, dtype=self.dtype)
        ctrl = relu_mpc.QReLUMPC(self.model, x0_lo, x0_up, u0_lo, u0_up)
        for i in range(10):
            x0 = torch.rand(self.vf.sys.x_dim, dtype=self.dtype) *\
                (x0_up - x0_lo) + x0_lo
            u_opt = self.vf_value_fun(x0)[1]
            if u_opt is None:
                continue
            u_opt = u_opt[:self.vf.sys.u_dim]
            u = ctrl.get_ctrl(x0)
            self.assertLessEqual(abs(u.item() - u_opt[0]), .2)


if __name__ == "__main__":
    unittest.main()
