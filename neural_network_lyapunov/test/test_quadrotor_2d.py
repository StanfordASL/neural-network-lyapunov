import neural_network_lyapunov.test.quadrotor_2d as quadrotor_2d

import unittest
import numpy as np
import torch
import scipy.integrate


class TestQuadrotor2D(unittest.TestCase):
    def test_dynamics_equilibrium(self):
        plant = quadrotor_2d.Quadrotor2D(torch.float64)
        u = plant.u_equilibrium
        xdot = plant.dynamics(np.zeros((6,)), u)
        np.testing.assert_allclose(xdot, np.zeros((6,)))

    def test_dynamics(self):
        plant = quadrotor_2d.Quadrotor2D(torch.float64)

        def check_dynamics(x, u):
            assert(isinstance(x, torch.Tensor))
            xdot = plant.dynamics(x, u)
            xdot_np = plant.dynamics(x.detach().numpy(), u.detach().numpy())
            np.testing.assert_allclose(xdot_np, xdot.detach().numpy())

        check_dynamics(
            torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float64),
            torch.tensor([7, 8], dtype=torch.float64))
        check_dynamics(
            torch.tensor([1, -2, 3, -4, 5, -6], dtype=torch.float64),
            torch.tensor([7, -8], dtype=torch.float64))

    def test_linearized_dynamics(self):
        plant = quadrotor_2d.Quadrotor2D(torch.float64)

        def check_linearized_dynamics(x, u):
            assert(isinstance(x, torch.Tensor))
            A, B = plant.linearized_dynamics(x, u)
            xdot = plant.dynamics(x, u)
            for i in range(6):
                if x.grad is not None:
                    x.grad.zero_()
                if u.grad is not None:
                    u.grad.zero_()
                xdot[i].backward(retain_graph=True)
                Ai_expected = x.grad.detach().numpy() if x.grad is not None\
                    else np.zeros((6,))
                np.testing.assert_allclose(
                    A[i, :].detach().numpy(), Ai_expected)
                Bi_expected = u.grad.detach().numpy() if u.grad is not None\
                    else np.zeros((2,))
                np.testing.assert_allclose(
                    B[i, :].detach().numpy(), Bi_expected)
            # Make sure numpy and torch input give same result.
            A_np, B_np = plant.linearized_dynamics(
                x.detach().numpy(), u.detach().numpy())
            np.testing.assert_allclose(A_np, A.detach().numpy())
            np.testing.assert_allclose(B_np, B.detach().numpy())

        check_linearized_dynamics(torch.tensor(
            [1, 2, 3, 4, 5, 6], dtype=torch.float64, requires_grad=True),
            torch.tensor([7, 8], dtype=torch.float64, requires_grad=True))
        check_linearized_dynamics(torch.tensor(
            [-1, -2, 3, 4, 5, 6], dtype=torch.float64, requires_grad=True),
            torch.tensor([7, -8], dtype=torch.float64, requires_grad=True))

    def test_lqr_control(self):
        plant = quadrotor_2d.Quadrotor2D(torch.float64)
        x_star = np.zeros((6,))
        u_star = plant.u_equilibrium.detach().numpy()
        Q = np.diag([10, 10, 10, 1, 1, plant.length/2./np.pi])
        R = np.array([[0.1, 0.05], [0.05, 0.1]])
        K, S = plant.lqr_control(Q, R, x_star, u_star)
        result = scipy.integrate.solve_ivp(
            lambda t, x: plant.dynamics(x, K@(x-x_star) + u_star), (0, 10),
            np.array([0.1, 0.1, -0.1, 0.2, 0.2, -0.3]))
        np.testing.assert_allclose(result.y[:, -1], x_star, atol=1E-6)


if __name__ == "__main__":
    unittest.main()
