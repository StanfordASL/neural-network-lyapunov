import numpy as np
import torch


class Rocket:
    def __init__(self):
        self.length = 0.2
        self.mass = 0.5
        self.inertia = 0.005
        self.gravity = 9.81

    @property
    def hover_thrust(self):
        return self.mass * self.gravity

    def dynamics(self, x, u):
        assert (x.shape == (6,))
        assert (u.shape == (2,))
        q = x[:3]
        qdot = x[3:]
        if isinstance(x, np.ndarray):
            c_theta = np.cos(q[2])
            s_theta = np.sin(q[2])
            qddot = np.array([
                (c_theta * u[0] - s_theta * u[1]) / self.mass,
                (s_theta * u[0] + c_theta * u[1]) / self.mass - self.gravity,
                self.length / 2 * u[0] / self.inertia
            ])
            return np.concatenate((qdot, qddot))
        elif isinstance(x, torch.Tensor):
            c_theta = torch.cos(q[2])
            s_theta = torch.sin(q[2])
            qddot = torch.stack((
                (c_theta * u[0] - s_theta * u[1]) / self.mass,
                (s_theta * u[0] + c_theta * u[1]) / self.mass - self.gravity,
                self.length / 2 * u[0] / self.inertia))
            return torch.cat((qdot, qddot))

    def linearized_dynamics(self, x, u):
        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        c_theta = np.cos(x[2])
        s_theta = np.sin(x[2])
        A[0:3, 3:6] = np.eye(3)
        A[3, 2] = (-s_theta * u[0] - c_theta * u[1]) / self.mass
        A[4, 2] = (c_theta * u[0] - s_theta * u[1]) / self.mass
        B[3, 0] = c_theta / self.mass
        B[3, 1] = -s_theta / self.mass
        B[4, 0] = s_theta / self.mass
        B[4, 1] = c_theta / self.mass
        B[5, 0] = self.length / (2 * self.inertia)
        return A, B
