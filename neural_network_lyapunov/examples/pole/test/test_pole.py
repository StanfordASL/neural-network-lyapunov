import neural_network_lyapunov.examples.pole.pole as mut
import neural_network_lyapunov.utils as utils

import torch
import numpy as np
import scipy.integrate
import unittest


class TestPole(unittest.TestCase):
    def test_dynamics(self):
        dut = mut.Pole(2., 5, 1.5)
        x_samples = utils.uniform_sample_in_box(
            torch.tensor([-1, -1, -2, -2, -2, -3, -3], dtype=dut.dtype),
            torch.tensor([1, 1, 2, 2, 2, 3, 3], dtype=dut.dtype), 100)
        u_samples = utils.uniform_sample_in_box(
            torch.tensor([-2, -3, -4], dtype=dut.dtype),
            torch.tensor([2, 3, 4], dtype=dut.dtype), 100)
        for i in range(x_samples.shape[0]):
            xdot_torch = dut.dynamics(x_samples[i], u_samples[i])
            xdot_np = dut.dynamics(x_samples[i].detach().numpy(),
                                   u_samples[i].detach().numpy())
            np.testing.assert_allclose(xdot_torch.detach().numpy(), xdot_np)

        # now simulate the pole system under zero external force. The total
        # energy should be conserved.
        # The state x0 is [x/y/z/ position of the end-effector, pole_state]
        x0 = np.array([0, 0, 0, 0.2, 0.1, 0.4, 0.5, -0.3, -0.1, 0.2])
        result = scipy.integrate.solve_ivp(
            lambda t, x: np.concatenate(
                (x[5:8], dut.dynamics(x[3:], np.zeros((3, ))))), [0, 1], x0)

        def total_energy(x, zA):
            x_AB = x[0]
            y_AB = x[1]
            z_AB = np.sqrt(dut.length**2 - x_AB**2 - y_AB**2)
            xdot_A = x[2]
            ydot_A = x[3]
            zdot_A = x[4]
            xdot_AB = x[5]
            ydot_AB = x[6]
            xdot_B = xdot_A + xdot_AB
            ydot_B = ydot_A + ydot_AB
            zdot_B = zdot_A - (x_AB * xdot_AB + y_AB * ydot_AB
                               ) / np.sqrt(dut.length**2 - x_AB**2 - y_AB**2)

            return 0.5 * dut.m_ee * (
                xdot_A**2 + ydot_A**2 + zdot_A**2
            ) + 0.5 * dut.m_sphere * (
                xdot_B**2 + ydot_B**2 + zdot_B**2
            ) + dut.m_ee * dut.gravity * zA + dut.m_sphere * dut.gravity * (
                zA + z_AB)

        self.assertAlmostEqual(total_energy(result.y[3:, 0], result.y[2, 0]),
                               total_energy(result.y[3:, -1], result.y[2, -1]),
                               places=1)

    def test_gradient(self):
        dut = mut.Pole(2., 5, 1.5)

        def compute_gradient_numerical(x, u):
            x_dim = x.shape[0]
            u_dim = u.shape[0]
            A = np.zeros((x_dim, x_dim))
            B = np.zeros((x_dim, u_dim))
            eps_val = 1E-7
            for i in range(x_dim):
                eps_x = np.zeros((x_dim, ))
                eps_x[i] = eps_val
                xdot_plus = dut.dynamics(x + eps_x, u)
                xdot_minus = dut.dynamics(x - eps_x, u)
                A[:, i] = (xdot_plus - xdot_minus) / (2 * eps_val)
            for i in range(u_dim):
                eps_u = np.zeros((u_dim, ))
                eps_u[i] = eps_val
                xdot_plus = dut.dynamics(x, u + eps_u)
                xdot_minus = dut.dynamics(x, u - eps_u)
                B[:, i] = (xdot_plus - xdot_minus) / (2 * eps_val)
            return A, B

        x_samples = utils.uniform_sample_in_box(
            torch.tensor([-1, -1, -2, -2, -2, -3, -3], dtype=dut.dtype),
            torch.tensor([1, 1, 2, 2, 2, 3, 3], dtype=dut.dtype), 10)
        u_samples = utils.uniform_sample_in_box(
            torch.tensor([-2, -3, -4], dtype=dut.dtype),
            torch.tensor([2, 3, 4], dtype=dut.dtype), 10)
        for i in range(x_samples.shape[0]):
            A, B = dut.gradient(x_samples[i], u_samples[i])
            A_expected, B_expected = compute_gradient_numerical(
                x_samples[i].detach().numpy(), u_samples[i].detach().numpy())
            np.testing.assert_allclose(A.detach().numpy(),
                                       A_expected,
                                       atol=1e-5)
            np.testing.assert_allclose(B.detach().numpy(),
                                       B_expected,
                                       atol=1e-5)


if __name__ == "__main__":
    unittest.main()
