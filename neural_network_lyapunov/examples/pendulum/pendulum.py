import torch

import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import gurobipy

import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


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
        A = torch.tensor(
            [[0, 1],
             [
                 -self.gravity / self.length * torch.cos(x[0]), -self.damping /
                 (self.mass * self.length * self.length)
             ]],
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
        A, B = self.dynamics_gradient(
            torch.tensor([np.pi, 0], dtype=self.dtype))
        S = scipy.linalg.solve_continuous_are(A.detach().numpy(),
                                              B.detach().numpy(), Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K


class PendulumVisualizer:
    def __init__(self, x0, figsize=(10, 10), subplot=111):
        """
        @param figsize The size of the fig
        @param subplot The argument in add_subplot(subplot) when adding the
        axis for pendulum.
        """
        self._plant = Pendulum(torch.float64)
        self._fig = plt.figure(figsize=figsize)
        self._pendulum_ax = self._fig.add_subplot(subplot)
        theta0 = x0[0]
        l_ = self._plant.length
        self._pendulum_arm, = self._pendulum_ax.plot(
            np.array([0, l_ * np.sin(theta0)]),
            np.array([0, -l_ * np.cos(theta0)]),
            linewidth=5)
        self._pendulum_sphere, = self._pendulum_ax.plot(l_ * np.sin(theta0),
                                                        -l_ * np.cos(theta0),
                                                        marker='o',
                                                        markersize=15)
        self._pendulum_ax.set_xlim(-l_ * 1.1, l_ * 1.1)
        self._pendulum_ax.set_ylim(-1.1 * l_, 1.1 * l_)
        self._pendulum_ax.set_axis_off()
        self._pendulum_title = self._pendulum_ax.set_title("t=0s")
        self._fig.canvas.draw()

    def draw(self, t, x):
        l_ = self._plant.length
        sin_theta = np.sin(x[0])
        cos_theta = np.cos(x[0])
        self._pendulum_arm.set_xdata(np.array([0, l_ * sin_theta]))
        self._pendulum_arm.set_ydata(np.array([0, -l_ * cos_theta]))
        self._pendulum_sphere.set_xdata(l_ * sin_theta)
        self._pendulum_sphere.set_ydata(-l_ * cos_theta)
        self._pendulum_title.set_text(f"t={t:.2f}s")
        self._fig.canvas.draw()


class PendulumReluContinuousTime:
    """
    The dynamics is theta_ddot = phi(theta, theta_dot, u) - phi(0, 0, 0)
    """
    def __init__(self, dtype, x_lo, x_up, u_lo, u_up, dynamics_relu):
        self.x_dim = 2
        self.dtype = dtype
        assert (x_lo.shape == (self.x_dim, ))
        assert (x_up.shape == (self.x_dim, ))
        self.x_lo = x_lo
        self.x_up = x_up
        self.u_dim = 1
        assert (u_lo.shape == (self.u_dim, ))
        assert (u_up.shape == (self.u_dim, ))
        self.u_lo = u_lo
        self.u_up = u_up
        assert (dynamics_relu[0].in_features == 3)
        assert (dynamics_relu[-1].out_features == 1)
        self.dynamics_relu = dynamics_relu
        self.x_equilibrium = torch.tensor([np.pi, 0], dtype=self.dtype)
        self.u_equilibrium = torch.tensor([0], dtype=self.dtype)
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

    def mixed_integer_constraints(
            self,
            u_lo=None,
            u_up=None) -> gurobi_torch_mip.MixedIntegerConstraintsReturn:
        if u_lo is None:
            u_lo = self.u_lo
        if u_up is None:
            u_up = self.u_up
        network_input_lo = torch.cat((self.x_lo, u_lo))
        network_input_up = torch.cat((self.x_up, u_up))
        result = self.dynamics_relu_free_pattern.output_constraint(
            network_input_lo, network_input_up,
            self.network_bound_propagate_method)
        # Add the constraint xdot[0] = x[1]
        # xdot[1] = phi(x, u) - phi(x*, u*)
        result.Cout = torch.cat(
            (torch.tensor([0], dtype=self.dtype),
             result.Cout[0] - self.dynamics_relu(
                 torch.cat((self.x_equilibrium, self.u_equilibrium)))))
        assert (result.Aout_input is None)
        result.Aout_input = torch.tensor([[0, 1, 0], [0, 0, 0]],
                                         dtype=self.dtype)
        result.Aout_slack = torch.cat((torch.zeros(
            (1, result.num_slack()), dtype=self.dtype), result.Aout_slack),
                                      dim=0)
        if (result.Aout_binary is None):
            result.Aout_binary = torch.zeros((2, result.num_binary()),
                                             dtype=self.dtype)
        else:
            result.Aout_binary = torch.cat(
                (torch.zeros((1, result.num_binary()),
                             dtype=self.dtype), result.Aout_binary),
                dim=0)
        relu_at_equilibrium = self.dynamics_relu(
            torch.cat((self.x_equilibrium, self.u_equilibrium)))
        result.x_next_lb = torch.stack(
            (self.x_lo[1], result.nn_output_lo[0] - relu_at_equilibrium[0]))
        result.x_next_ub = torch.stack(
            (self.x_up[1], result.nn_output_up[0] - relu_at_equilibrium[0]))
        return result

    def step_forward(self, x_start, u_start):
        if len(x_start.shape) == 1:
            theta_ddot = self.dynamics_relu(torch.cat(
                (x_start, u_start))) - self.dynamics_relu(
                    torch.cat((self.x_equilibrium, self.u_equilibrium)))
            return torch.stack((x_start[1], theta_ddot[0]))
        else:
            theta_ddot = self.dynamics_relu(
                torch.cat((x_start, u_start), dim=1)) - self.dynamics_relu(
                    torch.cat((self.x_equilibrium, self.u_equilibrium)))
            return torch.cat((x_start[:, 1:], theta_ddot), dim=1)

    def possible_dx(self, x, u):
        assert (isinstance(x, torch.Tensor))
        assert (isinstance(u, torch.Tensor))
        return [self.step_forward(x, u)]

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
