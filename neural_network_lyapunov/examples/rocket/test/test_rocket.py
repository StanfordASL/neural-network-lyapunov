import neural_network_lyapunov.examples.rocket.rocket as rocket

import unittest
import numpy as np
import torch
import scipy.integrate


class TestRocket(unittest.TestCase):
    def test_dynamics(self):
        plant = rocket.Rocket()
        plant.mass = 2

        def compute_xdot(x, u):
            c_theta = np.cos(x[2])
            s_theta = np.sin(x[2])
            qddot = np.zeros((3, ))
            qddot[0] = (c_theta * u[0] - s_theta * u[1]) / plant.mass
            qddot[1] = (s_theta * u[0] +
                        c_theta * u[1]) / plant.mass - plant.gravity
            qddot[2] = plant.length * u[0] / (2 * plant.inertia)
            return np.hstack((x[3:], qddot))

        x = np.array([3, 2., 1., 0.5, 0.4, 0.2])
        u = np.array([0.5, 2.5])
        np.testing.assert_allclose(plant.dynamics(x, u), compute_xdot(x, u))
        np.testing.assert_allclose(
            plant.dynamics(torch.from_numpy(x),
                           torch.from_numpy(u)).detach().numpy(),
            compute_xdot(x, u))
        x = np.array([-3, 2., -1., 1.3, 2.5, 0.4])
        u = np.array([-0.5, 2.5])
        np.testing.assert_allclose(plant.dynamics(x, u), compute_xdot(x, u))
        np.testing.assert_allclose(
            plant.dynamics(torch.from_numpy(x),
                           torch.from_numpy(u)).detach().numpy(),
            compute_xdot(x, u))
        x = np.array([-3, 2., -0.1, 0.4, 2.1, 2.5])
        u = np.array([-0.5, 5])
        np.testing.assert_allclose(plant.dynamics(x, u), compute_xdot(x, u))
        np.testing.assert_allclose(
            plant.dynamics(torch.from_numpy(x),
                           torch.from_numpy(u)).detach().numpy(),
            compute_xdot(x, u))

        x = np.array([1, 2., 0., 0, 0, 0])
        u = np.array([0, plant.hover_thrust])
        xdot = plant.dynamics(x, u)
        np.testing.assert_allclose(xdot, np.zeros((6, )))

    def test_linearized_dynamics(self):
        plant = rocket.Rocket()

        def torch_linearized_dynamics(x, u):
            x.requires_grad = True
            u.requires_grad = True
            A = torch.zeros((6, 6), dtype=torch.float64)
            B = torch.zeros((6, 2), dtype=torch.float64)
            for i in range(6):
                if x.grad is not None:
                    x.grad.zero_()
                if u.grad is not None:
                    u.grad.zero_()
                xdot = plant.dynamics(x, u)
                xdot[i].backward()
                A[i] = x.grad.clone()
                B[i] = u.grad.clone()
            return A, B

        def check(x, u):
            A, B = plant.linearized_dynamics(x, u)
            A_torch, B_torch = torch_linearized_dynamics(
                torch.from_numpy(x), torch.from_numpy(u))
            np.testing.assert_allclose(A_torch.detach().numpy(), A)
            np.testing.assert_allclose(B_torch.detach().numpy(), B)

        check(np.array([0.4, -0.1, 0.2, 0.5, 0.3, -0.2]), np.array([0.5, 2.1]))
        check(np.array([-0.4, -1.1, 1.2, 1.5, -0.3, 0.9]),
              np.array([-0.5, 2.1]))


class TestRocket2(unittest.TestCase):
    def test_dynamics(self):
        plant = rocket.Rocket()
        plant2 = rocket.Rocket2()

        def convert_rocket_state(x):
            """
            Given a state of Rocket (where the origin is in the center),
            convert it to the state of rocket2, where the origin is in the
            bottom of the rocket.
            """
            x2 = np.zeros_like(x)
            s_theta = np.sin(x[2])
            c_theta = np.cos(x[2])
            thetadot = x[5]
            x2[0] = x[0] + plant.length / 2 * s_theta
            x2[1] = x[1] - plant.length / 2 * c_theta
            x2[2] = x[2]
            x2[3] = x[3] + plant.length / 2 * c_theta * thetadot
            x2[4] = x[4] + plant.length / 2 * s_theta * thetadot
            x2[5] = thetadot
            return x2

        def test(x, u):
            result = scipy.integrate.solve_ivp(
                lambda t, x: plant.dynamics(x, u), (0, 0.01), x, atol=1E-7)
            x2 = convert_rocket_state(x)
            result2 = scipy.integrate.solve_ivp(
                lambda t, x: plant2.dynamics(x, u), (0, 0.01), x2, atol=1E-7)
            np.testing.assert_allclose(convert_rocket_state(result.y[:, -1]),
                                       result2.y[:, -1])
            xdot2 = plant2.dynamics(x, u)
            xdot2_torch = plant2.dynamics(torch.from_numpy(x),
                                          torch.from_numpy(u))
            np.testing.assert_allclose(xdot2, xdot2_torch.detach().numpy())

        test(np.array([0.2, 0.4, -0.1, 0.5, 0.3, 0.4]), np.array([0.5, 5.2]))
        test(np.array([1.2, 0.4, -0.5, 1.5, 0.7, -1.5]), np.array([-0.5,
                                                                   10.1]))
        test(np.array([1.2, 0.4, -0.5, 2.5, -0.7, -0.5]), np.array([-0.8,
                                                                    3.1]))

    def test_linearized_dynamics(self):
        plant = rocket.Rocket2()

        def torch_linearized_dynamics(x, u):
            x.requires_grad = True
            u.requires_grad = True
            A = torch.zeros((6, 6), dtype=torch.float64)
            B = torch.zeros((6, 2), dtype=torch.float64)
            for i in range(6):
                if x.grad is not None:
                    x.grad.zero_()
                if u.grad is not None:
                    u.grad.zero_()
                xdot = plant.dynamics(x, u)
                xdot[i].backward()
                A[i] = x.grad.clone()
                B[i] = u.grad.clone()
            return A, B

        def check(x, u):
            A, B = plant.linearized_dynamics(x, u)
            A_torch, B_torch = torch_linearized_dynamics(
                torch.from_numpy(x), torch.from_numpy(u))
            np.testing.assert_allclose(A_torch.detach().numpy(), A)
            np.testing.assert_allclose(B_torch.detach().numpy(), B)

        check(np.array([0.4, -0.1, 0.2, 0.5, 0.3, -0.2]), np.array([0.5, 2.1]))
        check(np.array([-0.4, -1.1, 1.2, 1.5, -0.3, 0.9]),
              np.array([-0.5, 2.1]))


if __name__ == "__main__":
    unittest.main()
