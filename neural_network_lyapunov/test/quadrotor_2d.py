import numpy as np
import torch
import scipy.integrate


class Quadrotor2D:
    def __init__(self, dtype):
        # length of the rotor arm.
        self.length = 0.25
        # mass of the quadrotor.
        self.mass = 0.486
        # moment of inertia
        self.inertia = 0.00383
        # gravity.
        self.gravity = 9.81
        self.dtype = dtype

    def dynamics(self, x, u):
        """
        Compute the continuous-time dynamics
        """
        q = x[:3]
        qdot = x[3:]
        if isinstance(x, np.ndarray):
            qddot = np.array([
                -np.sin(q[2]) / self.mass * (u[0] + u[1]),
                np.cos(x[2]) / self.mass * (u[0] + u[1]) - self.gravity,
                self.length / self.inertia * (u[0] - u[1])
            ])
            return np.concatenate((qdot, qddot))
        elif isinstance(x, torch.Tensor):
            qddot = torch.stack((
                -torch.sin(q[2]) / self.mass * (u[0] + u[1]),
                torch.cos(x[2]) / self.mass * (u[0] + u[1]) - self.gravity,
                self.length / self.inertia * (u[0] - u[1])))
            return torch.cat((qdot, qddot))

    def linearized_dynamics(self, x, u):
        """
        Return ∂ẋ/∂x and ∂ẋ/∂ u
        """
        if isinstance(x, np.ndarray):
            A = np.zeros((6, 6))
            B = np.zeros((6, 2))
            A[:3, 3:6] = np.eye(3)
            theta = x[2]
            A[3, 2] = -np.cos(theta) / self.mass * (u[0] + u[1])
            A[4, 2] = -np.sin(theta) / self.mass * (u[0] + u[1])
            B[3, 0] = -np.sin(theta) / self.mass
            B[3, 1] = B[3, 0]
            B[4, 0] = np.cos(theta) / self.mass
            B[4, 1] = B[4, 0]
            B[5, 0] = self.length / self.inertia
            B[5, 1] = -B[5, 0]
            return A, B
        elif isinstance(x, torch.Tensor):
            dtype = x.dtype
            A = torch.zeros((6, 6), dtype=dtype)
            B = torch.zeros((6, 2), dtype=dtype)
            A[:3, 3:6] = torch.eye(3, dtype=dtype)
            theta = x[2]
            A[3, 2] = -torch.cos(theta) / self.mass * (u[0] + u[1])
            A[4, 2] = -torch.sin(theta) / self.mass * (u[0] + u[1])
            B[3, 0] = -torch.sin(theta) / self.mass
            B[3, 1] = B[3, 0]
            B[4, 0] = torch.cos(theta) / self.mass
            B[4, 1] = B[4, 0]
            B[5, 0] = self.length / self.inertia
            B[5, 1] = -B[5, 0]
            return A, B

    @property
    def u_equilibrium(self):
        return torch.full(
            (2,), (self.mass * self.gravity) / 2, dtype=self.dtype)

    def lqr_control(self, Q, R, x, u):
        """
        The control action should be u = K * (x - x*) + u*
        """
        x_np = x if isinstance(x, np.ndarray) else x.detach().numpy()
        u_np = u if isinstance(u, np.ndarray) else u.detach().numpy()
        A, B = self.linearized_dynamics(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K, S


class Quadrotor2DVisualizer:
    """
    Copied from
    https://github.com/RussTedrake/underactuated/blob/master/underactuated/quadrotor2d.py
    """

    def __init__(self, ax, x_lim, y_lim):
        self.ax = ax
        self.ax.set_aspect("equal")
        self.ax.set_xlim(x_lim[0], x_lim[1])
        self.ax.set_ylim(y_lim[0], y_lim[1])

        self.length = .25  # moment arm (meters)

        self.base = np.vstack((1.2 * self.length * np.array([1, -1, -1, 1, 1]),
                               0.025 * np.array([1, 1, -1, -1, 1])))
        self.pin = np.vstack((0.005 * np.array([1, 1, -1, -1, 1]),
                              .1 * np.array([1, 0, 0, 1, 1])))
        a = np.linspace(0, 2 * np.pi, 50)
        self.prop = np.vstack(
            (self.length / 1.5 * np.cos(a), .1 + .02 * np.sin(2 * a)))

        # yapf: disable
        self.base_fill = self.ax.fill(
            self.base[0, :], self.base[1, :], zorder=1, edgecolor="k",
            facecolor=[.6, .6, .6])
        self.left_pin_fill = self.ax.fill(
            self.pin[0, :], self.pin[1, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 0])
        self.right_pin_fill = self.ax.fill(
            self.pin[0, :], self.pin[1, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 0])
        self.left_prop_fill = self.ax.fill(
            self.prop[0, :], self.prop[0, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 1])
        self.right_prop_fill = self.ax.fill(
            self.prop[0, :], self.prop[0, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 1])
        # yapf: enable

    def draw(self, t, x):
        R = np.array([[np.cos(x[2]), -np.sin(x[2])],
                      [np.sin(x[2]), np.cos(x[2])]])

        p = np.dot(R, self.base)
        self.base_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.base_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, np.vstack(
            (-self.length + self.pin[0, :], self.pin[1, :])))
        self.left_pin_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.left_pin_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]
        p = np.dot(R, np.vstack(
            (self.length + self.pin[0, :], self.pin[1, :])))
        self.right_pin_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.right_pin_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, np.vstack((
            -self.length + self.prop[0, :], self.prop[1, :])))
        self.left_prop_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.left_prop_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, np.vstack(
            (self.length + self.prop[0, :], self.prop[1, :])))
        self.right_prop_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.right_prop_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        self.ax.set_title("t = {:.1f}".format(t))
