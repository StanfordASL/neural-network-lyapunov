import neural_network_lyapunov.control_lyapunov as mut

import unittest
import torch
import numpy as np

import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.utils as utils


class TestControlLyapunov(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64

        # Create a linear system.
        A = torch.tensor([[1, 2], [-2, 3]], dtype=self.dtype)
        B = torch.tensor([[1, 3, 1], [0, 1, 0]], dtype=self.dtype)
        self.linear_system = control_affine_system.LinearSystem(
            A,
            B,
            x_lo=torch.tensor([-2, -3], dtype=self.dtype),
            x_up=torch.tensor([3, 3], dtype=self.dtype),
            u_lo=torch.tensor([-1, -2, -3], dtype=self.dtype),
            u_up=torch.tensor([1, 2, 3], dtype=self.dtype))
        self.lyapunov_relu1 = utils.setup_relu((2, 4, 3, 1),
                                               params=None,
                                               negative_slope=0.1,
                                               bias=True,
                                               dtype=self.dtype)
        self.lyapunov_relu1[0].weight.data = torch.tensor(
            [[1, -1], [0, 2], [-1, 2], [-2, 1]], dtype=self.dtype)
        self.lyapunov_relu1[0].bias.data = torch.tensor([1, -1, 0, 2],
                                                        dtype=self.dtype)
        self.lyapunov_relu1[2].weight.data = torch.tensor(
            [[3, -2, 1, 0], [1, -1, 2, 3], [-2, -1, 0, 3]], dtype=self.dtype)
        self.lyapunov_relu1[2].bias.data = torch.tensor([1, -2, 3],
                                                        dtype=self.dtype)
        self.lyapunov_relu1[4].weight.data = torch.tensor([[1, 3, -2]],
                                                          dtype=self.dtype)
        self.lyapunov_relu1[4].bias.data = torch.tensor([2], dtype=self.dtype)

    def lyapunov_derivative_tester(self, dut, x, x_equilibrium, V_lambda,
                                   epsilon, R):
        vdot = dut.lyapunov_derivative(x,
                                       x_equilibrium,
                                       V_lambda,
                                       epsilon,
                                       R=R)

        dphi_dx = utils.relu_network_gradient(dut.lyapunov_relu, x).squeeze(1)
        dl1_dx = V_lambda * utils.l1_gradient(R @ (x - x_equilibrium)) @ R
        v = dut.lyapunov_value(x, x_equilibrium, V_lambda, R=R)
        vdot_expected = np.inf
        for i in range(dphi_dx.shape[0]):
            for j in range(dl1_dx.shape[0]):
                dV_dx = dphi_dx[i] + dl1_dx[j]
                vdot_expected_ij = dV_dx @ dut.system.f(x) + epsilon * v
                dV_dx_times_G = dV_dx @ dut.system.G(x)
                for k in range(dut.system.u_dim):
                    vdot_expected_ij += torch.minimum(
                        dV_dx_times_G[k] * dut.system.u_lo[k],
                        dV_dx_times_G[k] * dut.system.u_up[k])
                if vdot_expected_ij < vdot_expected:
                    vdot_expected = vdot_expected_ij
        self.assertAlmostEqual(vdot.item(), vdot_expected.item())

    def test_lyapunov_derivative1(self):
        # Test with linear system.
        dut = mut.ControlLyapunov(self.linear_system, self.lyapunov_relu1)
        x_equilibrium = torch.tensor([0, -1], dtype=self.dtype)
        V_lambda = 0.5
        R = torch.tensor([[1, 0], [0, 2], [1, 3]], dtype=self.dtype)
        epsilon = 0.1

        x = torch.tensor([0.5, 1.5], dtype=self.dtype)
        self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                        epsilon, R)
        # Some ReLU inputs equal to 0.
        x = torch.tensor([1, 2], dtype=self.dtype)
        self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                        epsilon, R)
        # Some l1-norm entry equals to 0.
        x = torch.tensor([0, 1], dtype=self.dtype)
        self.lyapunov_derivative_tester(dut, x, x_equilibrium, V_lambda,
                                        epsilon, R)


if __name__ == "__main__":
    unittest.main()
