import numpy as np
import torch


class Rocket:
    """
    This rocket has state [x, z, theta, xdot, zdot, thetadot], where x/z are
    the position or the center of the rocket. theta is the angle between the
    rocket and the verticle line.
    """
    def __init__(self):
        self.length = 0.2
        self.mass = 0.5
        self.inertia = 0.005
        self.gravity = 9.81

    @property
    def hover_thrust(self):
        return self.mass * self.gravity

    def dynamics(self, x, u):
        assert (x.shape == (6, ))
        assert (u.shape == (2, ))
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
            qddot = torch.stack(
                ((c_theta * u[0] - s_theta * u[1]) / self.mass,
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


class Rocket2(Rocket):
    """
    This rocket has state [x, z, theta, xdot, zdot, thetadot], where x/z are
    the position or the BOTTOM of the rocket. theta is the angle between the
    rocket and the verticle line.
    """
    def __init__(self):
        super(Rocket2, self).__init__()

    def dynamics(self, x, u):
        assert (x.shape == (6, ))
        assert (u.shape == (2, ))
        if isinstance(x, np.ndarray):
            c_theta = np.cos(x[2])
            s_theta = np.sin(x[2])
            theta_dot = x[5]
            theta_ddot = self.length * u[0] / (2 * self.inertia)
            # The double time derivative of cos(theta)
            c_theta_ddot = -s_theta * theta_ddot - c_theta * (theta_dot**2)
            s_theta_ddot = c_theta * theta_ddot - s_theta * (theta_dot**2)
            xddot = (c_theta * u[0] - s_theta *
                     u[1]) / self.mass + self.length / 2 * s_theta_ddot
            zddot = (
                s_theta * u[0] + c_theta * u[1]
            ) / self.mass - self.gravity - self.length / 2 * c_theta_ddot
            return np.array([x[3], x[4], x[5], xddot, zddot, theta_ddot])
        elif isinstance(x, torch.Tensor):
            c_theta = torch.cos(x[2])
            s_theta = torch.sin(x[2])
            theta_dot = x[5]
            theta_ddot = self.length * u[0] / (2 * self.inertia)
            # The double time derivative of cos(theta)
            c_theta_ddot = -s_theta * theta_ddot - c_theta * (theta_dot**2)
            s_theta_ddot = c_theta * theta_ddot - s_theta * (theta_dot**2)
            xddot = (c_theta * u[0] - s_theta *
                     u[1]) / self.mass + self.length / 2 * s_theta_ddot
            zddot = (
                s_theta * u[0] + c_theta * u[1]
            ) / self.mass - self.gravity - self.length / 2 * c_theta_ddot
            return torch.stack((x[3], x[4], x[5], xddot, zddot, theta_ddot))

    def linearized_dynamics(self, x, u):
        assert (x.shape == (6, ))
        assert (u.shape == (2, ))
        s_theta = np.sin(x[2])
        c_theta = np.cos(x[2])
        thetadot = x[5]
        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        A[:3, 3:] = np.eye(3)
        A[3, 2] = (-s_theta * u[0] -
                   c_theta * u[1]) / self.mass + self.length / 2 * (
                       -s_theta * self.length * u[0] /
                       (2 * self.inertia) - c_theta * (thetadot**2))
        A[3, 5] = self.length / 2 * (-s_theta) * 2 * thetadot
        A[4, 2] = (c_theta * u[0] -
                   s_theta * u[1]) / self.mass + self.length / 2 * (
                       c_theta * self.length * u[0] /
                       (2 * self.inertia) - s_theta * (thetadot**2))
        A[4, 5] = self.length / 2 * c_theta * 2 * thetadot
        B[3, 0] = c_theta / self.mass + self.length**2 / 4 * c_theta / (
            self.inertia)
        B[3, 1] = -s_theta / self.mass
        B[4,
          0] = s_theta / self.mass + self.length**2 * s_theta / (4 *
                                                                 self.inertia)
        B[4, 1] = c_theta / self.mass
        B[5, 0] = self.length / (2 * self.inertia)
        return A, B


class RocketVisualizer:
    def __init__(self, ax, x_lim, y_lim, length):
        self.ax = ax
        self.ax.set_aspect("equal")
        self.ax.set_xlim(x_lim[0], x_lim[1])
        self.ax.set_ylim(y_lim[0], y_lim[1])

        self.length = length

        self.body = np.vstack(
            (self.length * np.array([-0.05, 0.05, 0.05, -0.05, -0.05]),
             self.length * np.array([-0.5, -0.5, 0.5, 0.5, -0.5])))
        self.head = np.vstack((self.length * np.array([-0.05, 0.05, 0, -0.05]),
                               self.length * np.array([0.5, 0.5, 0.6, 0.5])))
        self.bottom = np.vstack(
            (self.length * np.array([-0.05, -0.08, 0.08, 0.05, -0.05]),
             self.length * np.array([-0.5, -0.55, -0.55, -0.5, -0.5])))
        self.body_fill = self.ax.fill(self.body[0, :],
                                      self.body[1, :],
                                      zorder=1,
                                      edgecolor='k',
                                      facecolor=[.6, .6, .6])
        self.head_fill = self.ax.fill(self.head[0, :],
                                      self.head[1, :],
                                      zorder=0,
                                      edgecolor="k",
                                      facecolor=[0, 0, 0])
        self.bottom_fill = self.ax.fill(self.bottom[0, :],
                                        self.bottom[1, :],
                                        zorder=0,
                                        edgecolor="k",
                                        facecolor=[0, 0, 0])

    def draw(self, x):
        R = np.array([[np.cos(x[2]), -np.sin(x[2])],
                      [np.sin(x[2]), np.cos(x[2])]])
        p = np.dot(R, self.body)
        self.body_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.body_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, self.head)
        self.head_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.head_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, self.bottom)
        self.bottom_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.bottom_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]
