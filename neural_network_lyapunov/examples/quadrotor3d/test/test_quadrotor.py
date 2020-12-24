import neural_network_lyapunov.examples.quadrotor3d.quadrotor as quadrotor
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.test.test_relu_system as test_relu_system
import unittest
import numpy as np
import torch


class TestQuadrotorWithPixhawkReLUSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        torch.manual_seed(0)
        dynamics_relu = utils.setup_relu((10, 15, 10, 6),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=torch.float64)
        x_lo = torch.tensor(
            [-2, -2, -2, -0.5 * np.pi, -0.3 * np.pi, -np.pi, -3, -3, -3],
            dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.tensor([-0.5 * np.pi, -0.3 * np.pi, -np.pi, -5])
        u_up = -u_lo
        hover_thrust = 3.
        dt = 0.03
        self.dut = quadrotor.QuadrotorWithPixhawkReLUSystem(
            self.dtype, x_lo, x_up, u_lo, u_up, dynamics_relu, hover_thrust,
            dt)

        self.assertEqual(self.dut.x_dim, 9)
        self.assertEqual(self.dut.u_dim, 4)
        np.testing.assert_allclose(self.dut.u_equilibrium,
                                   np.array([0, 0, 0, hover_thrust]))

    def test_step_forward(self):
        def compute_next_state(x_val, u_val):
            pos_current = x_val[:3]
            rpy_current = x_val[3:6]
            vel_current = x_val[6:9]
            delta_vel_rpy = self.dut.dynamics_relu(
                torch.cat((vel_current, rpy_current,
                           u_val))) - self.dut.dynamics_relu(
                               torch.cat((torch.zeros(
                                   (6, ),
                                   dtype=self.dtype), self.dut.u_equilibrium)))
            vel_next = vel_current + delta_vel_rpy[:3]
            pos_next = pos_current + (vel_current + vel_next) * self.dut.dt / 2
            rpy_next = rpy_current + delta_vel_rpy[3:6]
            x_next_expected = torch.cat(
                (pos_next, rpy_next, vel_next)).detach().numpy()
            return x_next_expected

        # Test a single state/control
        x_start = torch.tensor([-0.2, 0.3, 0.4, -0.5, 1.2, 0.4, 0.3, 1.4, 0.9],
                               dtype=self.dtype)
        u_start = torch.tensor([-0.5, 1.2, 0.6, 1.3], dtype=self.dtype)
        x_next = self.dut.step_forward(x_start, u_start)
        np.testing.assert_allclose(x_next.detach().numpy(),
                                   compute_next_state(x_start, u_start))
        # Test a batch of state/control
        x_start = torch.tensor(
            [[-0.2, 0.1, 0.5, -1.2, -0.1, 0.9, 1.4, -0.3, 0.2],
             [0.5, -0.3, 1.2, -1.1, 0.8, 0.4, -0.5, 1.3, 1.2]],
            dtype=self.dtype)
        u_start = torch.tensor([[0.4, 0.3, -0.5, 1.1], [0.4, -0.3, 0.9, 1.3]],
                               dtype=self.dtype)
        x_next = self.dut.step_forward(x_start, u_start)
        for i in range(x_start.shape[0]):
            np.testing.assert_allclose(
                x_next[i].detach().numpy(),
                compute_next_state(x_start[i], u_start[i]))

    def test_add_dynamics_constraint(self):
        test_relu_system.check_add_dynamics_constraint(self.dut,
                                                       self.dut.x_equilibrium,
                                                       self.dut.u_equilibrium,
                                                       atol=1E-10)
        test_relu_system.check_add_dynamics_constraint(
            self.dut,
            torch.tensor([0.1, 0.5, -0.3, 0.2, 0.9, 1.1, 1.5, -1.1, 0.4],
                         dtype=self.dtype),
            torch.tensor([0.5, 0.8, -0.4, -1.2], dtype=self.dtype))
        test_relu_system.check_add_dynamics_constraint(
            self.dut,
            torch.tensor([-0.2, 0.7, 0.4, -1.2, 0.3, 0.4, -1.5, 1.3, 1.4],
                         dtype=self.dtype),
            torch.tensor([0.1, 0.9, 0.3, 1.5], dtype=self.dtype))


if __name__ == "__main__":
    unittest.main()
