import numpy as np
import torch


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
            qddot = torch.tensor([
                -np.sin(q[2]) / self.mass * (u[0] + u[1]),
                np.cos(x[2]) / self.mass * (u[0] + u[1]) - self.gravity,
                self.length / self.inertia * (u[0] - u[1])
            ], dtype=self.dtype)
            return torch.cat((qdot, qddot))


class Quadrotor2DVisualizer:
    """
    Copied from
    https://github.com/RussTedrake/underactuated/blob/master/underactuated/quadrotor2d.py
    """

    def __init__(self, ax, x_lim, y_lim):
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


class Quadrotor2DForwardSystem:
    """
    We use a forward network that maps (theta[n], u1[n], u2[n]) -> (ydot[n+1],
    zdot[n+1], thetadot[n+1]). Note that the forward
    network only considers the second-order dynamics, and because the dynamics
    is shift invariant and planar, the forward network input only considers the
    partial state, namely the orientation of the quadrotor.
    """
    def __init__(
        self, forward_network: torch.nn.Sequential, x_lo_all: torch.Tensor,
            x_up_all: torch.Tensor, dt: float):
        """
        @param forward_network A neural network, such that the dynamics is
        represented by [ydot[n+1], zdot[n+1], thetadot[n+1]] =
        phi(theta[n], u[n]) - phi(theta*, u*), where
        """
        assert(isinstance(forward_network, torch.nn.Sequential))
        assert(forward_network[0].in_features == 3)
        assert(forward_network[-1].out_features == 3)
        self._forward_network = forward_network
        self.dtype = torch.float64
        self.x_lo_all = x_lo_all
        self.x_up_all = x_up_all
        self.dt = dt

    @property
    def x_dim(self):
        return 6

    @property
    def u_dim(self):
        return 2

    @property
    def x_lo_all(self):
        return x_lo_all

    @property
    def x_up_all(self):
        return x_up_all

    def add_dynamics_constraint(
        self, mip, x_var, x_next_var, u_var, slack_var_name,
            binary_var_name):
        """
        Add the constraint between x[n], x[n+1] and u[n]
        """
        # First add the constraint that the network maps [theta[n], u1[n],
        # u2[n]] to [ydot[n+1], zdot[n+1], thetadot[n+1]]
        relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            self._forward_network, self.dtype)
        mip_cnstr_result, _, _, _, _ = relu_free_pattern.output_constraint(
            self.x_lo_all, self.x_up_all)
        slack, binary = mip.add_mixed_integer_linear_constraints(
            mip_cnstr_result, [x_var[2], u_var[0], u_var[1]], x_next_var[3:],
            slack_var_name, binary_var_name,
            "quadrotor2d_forward_dynamics_ineq",
            "quadrotor2d_forward_dynamics_eq",
            "quadrotor2d_forward_dynamics_output")
        # Add the constraint that q[n+1] - q[n] = (qdot[n+1] + qdot[n]) * dt/2
