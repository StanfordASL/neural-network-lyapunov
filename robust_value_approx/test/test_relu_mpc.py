import robust_value_approx.relu_mpc as relu_mpc
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.value_to_optimization as value_to_optimization
import double_integrator

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
        self.model = torch.load("double_integrator_model.pt")

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


if __name__ == "__main__":
    unittest.main()
