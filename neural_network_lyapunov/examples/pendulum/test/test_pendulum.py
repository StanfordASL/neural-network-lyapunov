import neural_network_lyapunov.examples.pendulum.pendulum as pendulum

import unittest

import torch
import numpy as np
import scipy.integrate


class TestPendulum(unittest.TestCase):
    def test_dynamics_gradient(self):
        dtype = torch.float64
        dut = pendulum.Pendulum(dtype)

        def test_gradient(x, u):
            xdot = dut.dynamics(x, u)
            (A_expected0, B_expected0) = torch.autograd.grad(
                xdot, (x, u),
                grad_outputs=torch.tensor([1, 0], dtype=dtype),
                retain_graph=True)
            (A_expected1, B_expected1) = torch.autograd.grad(
                xdot, (x, u),
                grad_outputs=torch.tensor([0, 1], dtype=dtype),
                retain_graph=True)
            A_expected = torch.cat((A_expected0.reshape(
                (1, -1)), A_expected1.reshape((1, -1))),
                                   dim=0)
            B_expected = torch.cat((B_expected0.reshape(
                (1, -1)), B_expected1.reshape((1, -1))),
                                   dim=0)
            A, B = dut.dynamics_gradient(x)
            np.testing.assert_allclose(A_expected.detach().numpy(),
                                       A.detach().numpy())
            np.testing.assert_allclose(B_expected.detach().numpy(),
                                       B.detach().numpy())

        test_gradient(torch.tensor([0, 1], dtype=dtype, requires_grad=True),
                      torch.tensor([2], dtype=dtype, requires_grad=True))
        test_gradient(torch.tensor([3, 1], dtype=dtype, requires_grad=True),
                      torch.tensor([2], dtype=dtype, requires_grad=True))
        test_gradient(torch.tensor([3, -1], dtype=dtype, requires_grad=True),
                      torch.tensor([2], dtype=dtype, requires_grad=True))

    def test_lqr_control(self):
        dtype = torch.float64
        dut = pendulum.Pendulum(dtype)

        Q = np.diag(np.array([1, 10.]))
        R = np.eye(1)
        K = dut.lqr_control(Q, R)

        # Now start with a state close to the [pi, 0], and simulate it with the
        # lqr controller.
        x_des = np.array([np.pi, 0])

        result = scipy.integrate.solve_ivp(
            lambda t, x: dut.dynamics(x, K @ (x - x_des)), (0, 5),
            np.array([np.pi + 0.05, 0.1]))
        np.testing.assert_allclose(result.y[:, -1], x_des, atol=2E-5)

    def test_swing_up_control(self):
        # We use a energy shaping controller and an LQR controller to swing up
        # the pendulum.
        plant = pendulum.Pendulum(torch.float64)
        Q = np.diag([1, 10])
        R = np.array([[1]])
        x_des = np.array([np.pi, 0])
        lqr_gain = plant.lqr_control(Q, R)

        def controller(x):
            if (x - x_des).dot(Q @ (x - x_des)) > 0.1:
                u = plant.energy_shaping_control(x, x_des, 10)
            else:
                u = lqr_gain @ (x - x_des)
            return u

        def converged(t, y):
            return np.linalg.norm(y - x_des) - 1E-3

        converged.terminal = True

        x0s = [np.array([0, 0.]), np.array([0.1, -1]), np.array([1.5, 0.5])]
        for x0 in x0s:
            result = scipy.integrate.solve_ivp(
                lambda t, x: plant.dynamics(x, controller(x)), (0, 10),
                x0,
                events=converged)
            np.testing.assert_allclose(result.y[:, -1], x_des, atol=1E-3)


if __name__ == "__main__":
    unittest.main()
