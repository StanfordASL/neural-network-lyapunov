import neural_network_lyapunov.test.quadrotor_2d as quadrotor_2d

import unittest
import numpy as np
import torch


class TestQuadrotor2D(unittest.TestCase):
    def test_dynamics_equilibrium(self):
        plant = quadrotor_2d.Quadrotor2D(torch.float64)
        u = np.full((2,), plant.mass * plant.gravity / 2)
        xdot = plant.dynamics(np.zeros((6,)), u)
        np.testing.assert_allclose(xdot, np.zeros((6,)))

if __name__ == "__main__":
    unittest.main()
