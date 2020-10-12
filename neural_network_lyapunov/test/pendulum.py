import torch

import numpy as np
import scipy
import scipy.linalg


class Pendulum:
    def __init__(self, dtype):
        self.dtype = dtype
        self.mass = 1
        self.gravity = 9.81
        self.length = 1
        self.damping = 0.1

    def dynamics(self, x, u):
        theta = x[0]
        thetadot = x[1]
        if isinstance(x, np.ndarray):
            thetaddot = (u[0] - self.mass * self.gravity * self.length *
                         np.sin(theta) - self.damping * thetadot) /\
                (self.mass * self.length * self.length)
            return np.array([thetadot, thetaddot])
        elif isinstance(x, torch.Tensor):
            thetaddot = (u[0] - self.mass * self.gravity * self.length *
                         torch.sin(theta) - self.damping * thetadot) /\
                (self.mass * self.length * self.length)
            return torch.cat((thetadot.view(1), thetaddot.view(1)))

    def potential_energy(self, x):
        if isinstance(x, torch.Tensor):
            cos_theta = torch.cos(x[0])
        elif isinstance(x, np.ndarray):
            cos_theta = np.cos(x[0])
        return -self.mass * self.gravity * self.length * cos_theta

    def kinetic_energy(self, x):
        if isinstance(x, torch.Tensor):
            l_thetadot_square = torch.pow(self.length * x[1], 2)
        elif isinstance(x, np.ndarray):
            l_thetadot_square = np.power(self.length * x[1], 2)
        return 0.5 * self.mass * l_thetadot_square

    def energy_shaping_control(self, x, x_des, gain):
        """
        The control law is u = -k*thetadot * (E - E_des)
        """
        E_des = self.potential_energy(x_des) + self.kinetic_energy(x_des)
        E = self.potential_energy(x) + self.kinetic_energy(x)
        if x[1] == 0:
            u = -gain * (E - E_des)
        else:
            u = -gain * x[1] * (E - E_des)
        return np.array([u])

    def dynamics_gradient(self, x):
        """
        Returns the gradient of the dynamics
        """
        A = torch.tensor([[0, 1], [
            -self.gravity / self.length * torch.cos(x[0]),
            -self.damping / (self.mass * self.length * self.length)]],
            dtype=self.dtype)
        B = torch.tensor([[0], [1 / (self.mass * self.length * self.length)]],
                         dtype=self.dtype)
        return A, B

    def lqr_control(self, Q, R):
        """
        lqr control around the equilibrium (pi, 0).
        returns the controller gain K
        The control action should be u = K * (x - x_des)
        """
        # First linearize the dynamics
        # The dynamics is
        # thetaddot = (u - mgl * sin(theta) - b*thetadot) / (ml^2)
        A, B = self.dynamics_gradient(torch.tensor(
            [np.pi, 0], dtype=self.dtype))
        S = scipy.linalg.solve_continuous_are(
            A.detach().numpy(), B.detach().numpy(), Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K
