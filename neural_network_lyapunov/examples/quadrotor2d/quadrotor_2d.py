import numpy as np
import torch
import scipy.integrate
import gurobipy

import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


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
            qddot = torch.stack(
                (-torch.sin(q[2]) / self.mass * (u[0] + u[1]),
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
        return torch.full((2, ), (self.mass * self.gravity) / 2,
                          dtype=self.dtype)

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

    def next_pose(self, x, u, dt):
        """
        Computes the next pose of the quadrotor after dt.
        """
        x_np = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        u_np = u.detach().numpy() if isinstance(u, torch.Tensor) else u
        result = scipy.integrate.solve_ivp(
            lambda t, x_val: self.dynamics(x_val, u_np), [0, dt], x_np)
        return result.y[:, -1]


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

        p = np.dot(
            R, np.vstack((-self.length + self.prop[0, :], self.prop[1, :])))
        self.left_prop_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.left_prop_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R,
                   np.vstack((self.length + self.prop[0, :], self.prop[1, :])))
        self.right_prop_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.right_prop_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        self.ax.set_title("t = {:.1f}".format(t))
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('z (m)')


class QuadrotorReluContinuousTime:
    """
    The dynamics is qddot = phi(theta, u) - phi(0, u*)
    """
    def __init__(self, dtype, x_lo, x_up, u_lo, u_up, dynamics_relu,
                 u_equilibrium: torch.Tensor):
        self.x_dim = 6
        self.dtype = dtype
        assert (x_lo.shape == (self.x_dim, ))
        assert (x_up.shape == (self.x_dim, ))
        self.x_lo = x_lo
        self.x_up = x_up
        self.u_dim = 2
        assert (u_lo.shape == (self.u_dim, ))
        assert (u_up.shape == (self.u_dim, ))
        self.u_lo = u_lo
        self.u_up = u_up
        assert (dynamics_relu[0].in_features == 3)
        assert (dynamics_relu[-1].out_features == 3)
        self.dynamics_relu = dynamics_relu
        self.x_equilibrium = torch.zeros((6, ), dtype=self.dtype)
        self.u_equilibrium = u_equilibrium
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        self.network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.IA

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def step_forward(self, x_start, u_start):
        relu_at_equilibrium = self.dynamics_relu(
            torch.cat((torch.tensor([0],
                                    dtype=self.dtype), self.u_equilibrium)))
        if len(x_start.shape) == 1:
            q_ddot = self.dynamics_relu(torch.cat(
                (x_start[2:3], u_start))) - relu_at_equilibrium
            return torch.cat((x_start[3:], q_ddot))
        else:
            q_ddot = self.dynamics_relu(
                torch.cat(
                    (x_start[:, 2:3], u_start), dim=1)) - relu_at_equilibrium
            return torch.cat((x_start[:, 3:], q_ddot), dim=1)

    def mixed_integer_constraints(
            self,
            u_lo=None,
            u_up=None) -> gurobi_torch_mip.MixedIntegerConstraintsReturn:
        if u_lo is None:
            u_lo = self.u_lo
        if u_up is None:
            u_up = self.u_up
        network_input_lo = torch.cat((self.x_lo[2:3], u_lo))
        network_input_up = torch.cat((self.x_up[2:3], u_up))
        result = self.dynamics_relu_free_pattern.output_constraint(
            network_input_lo, network_input_up,
            self.network_bound_propagate_method)
        # Change the input from [x, u] to [theta, u]
        A_transform = torch.zeros((3, 8), dtype=self.dtype)
        A_transform[0, 2] = 1
        A_transform[1, -2] = 1
        A_transform[2, -1] = 1
        result.transform_input(A_transform, torch.zeros((3, ),
                                                        dtype=self.dtype))
        # Add the constraint xdot[:3] = x[3:] and
        # xdot[3:] = phi(theta, u) - phi(0, u_equilibrium)
        relu_at_equilibrium = self.dynamics_relu(
            torch.cat((torch.tensor([0],
                                    dtype=self.dtype), self.u_equilibrium)))
        result.Cout = torch.cat((torch.zeros(
            (3, ), dtype=self.dtype), result.Cout - relu_at_equilibrium),
                                dim=0)
        assert (result.Aout_input is None)
        result.Aout_input = torch.zeros((6, 8), dtype=self.dtype)
        result.Aout_input[:3, 3:6] = torch.eye(3, dtype=self.dtype)
        result.Aout_slack = torch.cat((torch.zeros(
            (3, result.num_slack()), dtype=self.dtype), result.Aout_slack),
                                      dim=0)
        if result.Aout_binary is not None:
            result.Aout_binary = torch.cat(
                (torch.zeros((3, result.num_binary()),
                             dtype=self.dtype), result.Aout_binary),
                dim=0)

        # Add the constraint x_lo[:2] <= input[:2] <= x_up[:2]
        # x_lo[3:] <= input[3:6] <= x_up[3:]
        Ain_input_additional = torch.zeros((10, 8), dtype=self.dtype)
        Ain_input_additional[:2, :2] = torch.eye(2, dtype=self.dtype)
        Ain_input_additional[2:4, :2] = -torch.eye(2, dtype=self.dtype)
        Ain_input_additional[4:7, 3:6] = torch.eye(3, dtype=self.dtype)
        Ain_input_additional[7:10, 3:6] = -torch.eye(3, dtype=self.dtype)
        result.Ain_input = torch.cat((result.Ain_input, Ain_input_additional),
                                     dim=0)
        result.Ain_slack = torch.cat(
            (result.Ain_slack,
             torch.zeros((10, result.num_slack()), dtype=self.dtype)),
            dim=0)
        result.Ain_binary = torch.cat(
            (result.Ain_binary,
             torch.zeros((10, result.num_binary()), dtype=self.dtype)),
            dim=0)
        result.rhs_in = torch.cat(
            (result.rhs_in, self.x_up[:2], -self.x_lo[:2], self.x_up[3:6],
             -self.x_lo[3:6]))

        result.x_next_lb = torch.cat(
            (self.x_lo[3:], result.nn_output_lo - relu_at_equilibrium), dim=0)
        result.x_next_ub = torch.cat(
            (self.x_up[3:], result.nn_output_up - relu_at_equilibrium), dim=0)
        return result

    def add_dynamics_constraint(
        self,
        mip,
        x_var,
        x_next_var,
        u_var,
        slack_var_name,
        binary_var_name,
        additional_u_lo: torch.Tensor = None,
        additional_u_up: torch.Tensor = None,
        binary_var_type=gurobipy.GRB.BINARY,
        u_input_prog: relu_system.ControlBoundProg = None
    ) -> relu_system.ReLUDynamicsConstraintReturn:
        return relu_system._add_forward_dynamics_mip_constraints(
            self, mip, x_var, x_next_var, u_var, slack_var_name,
            binary_var_name, additional_u_lo, additional_u_up, binary_var_type,
            u_input_prog)
