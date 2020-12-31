import neural_network_lyapunov.examples.car.acceleration_car as\
    acceleration_car
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.test.test_relu_system as test_relu_system
import unittest
import numpy as np
import torch
import scipy.integrate


class TestAccelerationCar(unittest.TestCase):
    def test_dynamics(self):
        dtype = torch.float64
        plant = acceleration_car.AccelerationCar(dtype)
        np.testing.assert_allclose(
            plant.dynamics(np.zeros((4, )), np.zeros((2, ))), np.zeros((4, )))

        def tester(x_val, u_val):
            x = x_val.detach().numpy() if isinstance(x_val,
                                                     torch.Tensor) else x_val
            u = u_val.detach().numpy() if isinstance(u_val,
                                                     torch.Tensor) else x_val

            xdot = plant.dynamics(x, u)
            theta = x[2]
            vel = x[3]
            theta_dot = u[0]
            accel = u[1]
            pos_dot = np.array([vel * np.cos(theta), vel * np.sin(theta)])
            xdot_expected = np.array(
                [pos_dot[0], pos_dot[1], theta_dot, accel])
            np.testing.assert_allclose(xdot, xdot_expected)

        tester(np.array([1., -2., 3., 0.5]), np.array([0.4, -1.2]))
        tester(np.array([1.3, -1.2, 3.4, 1.5]), np.array([1.2, -1.9]))
        tester(torch.tensor([1.3, -1.2, 3.4, 1.5], dtype=dtype),
               torch.tensor([1.2, -1.9], dtype=dtype))

    def test_next_pose(self):
        x = np.array([0.4, 1.2, -0.5, 3.1])
        u = np.array([2.1, -0.5])
        dt = 0.01
        plant = acceleration_car.AccelerationCar(torch.float64)
        result = scipy.integrate.solve_ivp(
            lambda t, x_val: plant.dynamics(x_val, u), (0, dt), x)
        np.testing.assert_allclose(result.y[:3, -1], plant.next_pose(x, u, dt))


class TestAccelerationCarReLUSystem(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        torch.manual_seed(0)
        dynamics_relu = utils.setup_relu((4, 8, 8, 2),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=self.dtype)
        x_lo = torch.tensor([-5, -5, -np.pi, -4], dtype=self.dtype)
        x_up = -x_lo
        u_lo = torch.tensor([-2 * np.pi, -6], dtype=self.dtype)
        u_up = -u_lo
        dt = 0.01
        self.dut = acceleration_car.AccelerationCarReLUModel(
            self.dtype, x_lo, x_up, u_lo, u_up, dynamics_relu, dt)
        self.assertEqual(self.dut.x_dim, 4)
        self.assertEqual(self.dut.u_dim, 2)
        np.testing.assert_allclose(self.dut.x_equilibrium, np.zeros((4, )))
        np.testing.assert_allclose(self.dut.u_equilibrium, np.zeros((2, )))

    def test_step_forward(self):
        np.testing.assert_allclose(
            self.dut.step_forward(self.dut.x_equilibrium,
                                  self.dut.u_equilibrium).detach().numpy(),
            self.dut.x_equilibrium.detach().numpy())

        def compute_next_state(x, u):
            delta_pos = self.dut.dynamics_relu(torch.cat(
                (x[2:], u))) - self.dut.dynamics_relu(
                    torch.zeros((4, ), dtype=self.dtype))
            pos_next = x[:2] + delta_pos
            theta_next = x[2] + u[0] * self.dut.dt
            vel_next = x[3] + u[1] * self.dut.dt
            return torch.cat((pos_next, torch.stack((theta_next, vel_next))))

        x = torch.tensor([0.5, 0.3, 0.2, -0.9], dtype=self.dtype)
        u = torch.tensor([1.2, 0.4], dtype=self.dtype)
        np.testing.assert_allclose(
            self.dut.step_forward(x, u).detach().numpy(),
            compute_next_state(x, u).detach().numpy())
        x = torch.tensor([-2.5, 1.3, 0.9, -2.9], dtype=self.dtype)
        u = torch.tensor([1.2, -3.4], dtype=self.dtype)
        np.testing.assert_allclose(
            self.dut.step_forward(x, u).detach().numpy(),
            compute_next_state(x, u).detach().numpy())

        x = torch.tensor([[0.5, 1.2, 2.1, 0.5], [0.4, 3.2, -0.5, -2.1]],
                         dtype=self.dtype)
        u = torch.tensor([[0.4, 0.3], [1.2, -0.6]], dtype=self.dtype)
        x_next = self.dut.step_forward(x, u)
        for i in range(x.shape[0]):
            np.testing.assert_allclose(
                x_next[i].detach().numpy(),
                compute_next_state(x[i], u[i]).detach().numpy())

    def test_add_dynamics_constraint(self):
        test_relu_system.check_add_dynamics_constraint(
            self.dut,
            x_val=torch.tensor([-2.5, -3.1, 0.8, -2.4], dtype=self.dtype),
            u_val=torch.tensor([0.5, 0.9], dtype=self.dtype))
        test_relu_system.check_add_dynamics_constraint(
            self.dut,
            x_val=torch.tensor([1.5, -2.1, -1.8, 2.8], dtype=self.dtype),
            u_val=torch.tensor([-1.5, 2.9], dtype=self.dtype))


if __name__ == "__main__":
    unittest.main()
