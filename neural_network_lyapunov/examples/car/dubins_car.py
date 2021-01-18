import torch
import numpy as np
import scipy.integrate

import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.mip_utils as mip_utils

import gurobipy


class DubinsCar:
    """
    A simple Dubin's car model, that the state is [pos_x, pos_y, theta], and
    the control is [vel, thetadot], where vel is the velocity along the heading
    direction of the car. theta is the counter clockwise angle (yaw angle) of
    the car from +x axis.
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def dynamics(self, x, u):
        """
        Compute xdot of the Dubins car.
        """
        theta = x[2]
        vel = u[0]
        thetadot = u[1]
        if isinstance(x, np.ndarray):
            return np.array(
                [vel * np.cos(theta), vel * np.sin(theta), thetadot])
        elif isinstance(x, torch.Tensor):
            return torch.stack(
                (vel * torch.cos(theta), vel * torch.sin(theta), thetadot))

    def dynamics_gradient(self, x, u):
        """
        Compute the gradient A = ∂f/∂x, B = ∂f/∂u
        """
        theta = x[2]
        vel = u[0]
        if isinstance(x, np.ndarray):
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            return np.array([[0, 0, -vel * sin_theta], [0, 0, vel * cos_theta],
                             [0, 0, 0]]), np.array([[cos_theta, 0],
                                                    [sin_theta, 0], [0., 1]])
        elif isinstance(x, torch.Tensor):
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            A = torch.zeros((3, 3), dtype=x.dtype)
            B = torch.zeros((3, 2), dtype=x.dtype)
            A[0, 2] = -vel * sin_theta
            A[1, 2] = vel * cos_theta
            B[0, 0] = cos_theta
            B[1, 0] = sin_theta
            B[2, 1] = 1
            return A, B

    def next_pose(self, x, u, dt):
        """
        Computes the next pose of the car after dt.
        """
        x_np = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        u_np = u.detach().numpy() if isinstance(u, torch.Tensor) else u
        result = scipy.integrate.solve_ivp(
            lambda t, x_val: self.dynamics(x_val, u_np), [0, dt], x_np)
        return result.y[:, -1]


class DubinsCarReLUModel:
    """
    We model the discrete-time dynamics of Dubins car using a multi-layer
    perceptron with (leaky) ReLU units.
    We consider the origin being the equilibrium state of the Dubins car. The
    neural network predicts the next state as the following
    [delta_pos_x, delta_pos_y] = ϕ(θ, vel, θ_dot) - ϕ(θ, 0, θ_dot)
    where vel is the velocity along the car heading direction.
    """
    def __init__(self, dtype, x_lo: torch.Tensor, x_up: torch.Tensor,
                 u_lo: torch.Tensor, u_up: torch.Tensor,
                 dynamics_relu: torch.nn.Sequential, dt: float,
                 thetadot_as_input: bool):
        """
        @param x_lo The lower bound of the state.
        @param x_up The upper bound of the state.
        @param u_lo The lower input limits.
        @param u_up The upper input limits.
        @param dynamics_relu ϕ in the documentation above.
        @param dt The delta time (which is also used in training the dynamics
        relu).
        @param thetadot_as_input A boolean flag, if set to True, then the
        network input is [θ, vel, θ_dot]; otherwise the network input is
        [θ, vel]
        """
        self.dtype = dtype
        self.x_dim = 3
        assert (isinstance(x_lo, torch.Tensor))
        assert (x_lo.shape == (3, ))
        self.x_lo = x_lo
        assert (isinstance(x_up, torch.Tensor))
        self.x_up = x_up
        self.u_dim = 2
        assert (isinstance(u_lo, torch.Tensor))
        assert (u_lo.shape == (2, ))
        self.u_lo = u_lo
        assert (isinstance(u_up, torch.Tensor))
        assert (u_up.shape == (2, ))
        self.u_up = u_up
        assert (isinstance(thetadot_as_input, bool))
        self.thetadot_as_input = thetadot_as_input
        if thetadot_as_input:
            assert (dynamics_relu[0].in_features == 3)
        else:
            assert (dynamics_relu[0].in_features == 2)
        assert (dynamics_relu[-1].out_features == 2)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        self.x_equilibrium = torch.zeros((3, ), dtype=dtype)
        self.u_equilibrium = torch.zeros((2, ), dtype=dtype)
        assert (isinstance(dt, float))
        assert (dt > 0)
        self.dt = dt

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def step_forward(self, x_start: torch.Tensor, u_start: torch.Tensor) ->\
            torch.Tensor:
        """
        Compute x[n+1] according to
        θ[n+1] = θ[n] + θ̇_dot[n]*dt
        when thetadot_as_input is true
        [pos_x[n+1], pos_y[n+1]] =
        [pos_x[n], pos_y[n]] + ϕ(θ[n], vel[n], θ_dot[n]) − ϕ(θ[n], 0, θ_dot[n])
        else
        [pos_x[n+1], pos_y[n+1]] =
        [pos_x[n], pos_y[n]] + ϕ(θ[n], vel[n]) − ϕ(θ[n], 0)
        """
        assert (isinstance(x_start, torch.Tensor))
        assert (isinstance(u_start, torch.Tensor))
        if len(x_start.shape) == 1:
            # single data point.
            assert (torch.all(u_start <= self.u_up))
            assert (torch.all(u_start >= self.u_lo))
            if self.thetadot_as_input:
                network_input = torch.stack(
                    (x_start[2], u_start[0], u_start[1]))
                network_input_zero = torch.stack(
                    (x_start[2], torch.tensor(0,
                                              dtype=self.dtype), u_start[1]))
            else:
                network_input = torch.stack((x_start[2], u_start[0]))
                network_input_zero = torch.stack(
                    (x_start[2], torch.tensor(0, dtype=self.dtype)))
            delta_position = self.dynamics_relu(
                network_input) - self.dynamics_relu(network_input_zero)
            position_next = x_start[:2] + delta_position
            theta_next = x_start[2] + u_start[1] * self.dt
            return torch.cat((position_next, theta_next.reshape((-1, ))))
        elif len(x_start.shape) == 2:
            # batch of data
            if self.thetadot_as_input:
                network_input = torch.cat((x_start[:, 2].reshape(
                    (-1, 1)), u_start),
                                          dim=1)
                network_input_zero = torch.cat(
                    (x_start[:, 2].reshape((-1, 1)),
                     torch.zeros((x_start.shape[0], 1),
                                 dtype=self.dtype), u_start[:, 1].reshape(
                                     (-1, 1))),
                    dim=1)
            else:
                network_input = torch.cat((x_start[:, 2].reshape(
                    (-1, 1)), u_start[:, 0].reshape((-1, 1))),
                                          dim=1)
                network_input_zero = torch.cat(
                    (x_start[:, 2].reshape((-1, 1)),
                     torch.zeros((x_start.shape[0], 1), dtype=self.dtype)),
                    dim=1)
            delta_position = self.dynamics_relu(
                network_input) - self.dynamics_relu(network_input_zero)
            position_next = x_start[:, :2] + delta_position
            theta_next = (x_start[:, 2] + u_start[:, 1] * self.dt).reshape(
                (-1, 1))
            return torch.cat((position_next, theta_next), dim=1)

    def possible_dx(self, x, u):
        """
        TODO(hongkai.dai): I will deprecate this function soon. But for now I
        keep it for backward maintanence as other systems in relu_system.py
        """
        assert (isinstance(x, torch.Tensor))
        assert (isinstance(u, torch.Tensor))
        return [self.step_forward(x, u)]

    def add_dynamics_constraint(self, mip: gurobi_torch_mip.GurobiTorchMIP,
                                x_var, x_next_var, u_var, slack_var_name,
                                binary_var_name):
        """
        Add the dynamic constraints a mixed-integer linear constraints. Refer
        to relu_system.py for the common API.
        The constraints are
        θ[n+1] = θ[n] + θ̇_dot[n]*dt
        when thetadot_as_input is true
        [pos_x[n+1], pos_y[n+1]] =
        [pos_x[n], pos_y[n]] + ϕ(θ[n], vel[n], θ_dot[n]) − ϕ(θ[n], 0, θ_dot[n])
        else
        [pos_x[n+1], pos_y[n+1]] =
        [pos_x[n], pos_y[n]] + ϕ(θ[n], vel[n]) − ϕ(θ[n], 0)
        Note that the network ϕ takes two different set of inputs, one is
        (θ[n], vel[n], θ_dot[n]), another one is (θ[n], 0, θ_dot[n])
        """
        assert (isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
        if self.thetadot_as_input:
            network_input_lo = torch.stack(
                (self.x_lo[2], self.u_lo[0], self.u_lo[1]))
            network_input_up = torch.stack(
                (self.x_up[2], self.u_up[0], self.u_up[1]))
            # The lower/upper bound for the (theta, 0, thetadot)
            network_input_zero_vel_lo = torch.stack(
                (self.x_lo[2], torch.tensor(0,
                                            dtype=self.dtype), self.u_lo[1]))
            network_input_zero_vel_up = torch.stack(
                (self.x_up[2], torch.tensor(0,
                                            dtype=self.dtype), self.u_up[1]))
        else:
            network_input_lo = torch.stack((self.x_lo[2], self.u_lo[0]))
            network_input_up = torch.stack((self.x_up[2], self.u_up[0]))
            network_input_zero_vel_lo = torch.stack(
                (self.x_lo[2], torch.tensor(0, dtype=self.dtype)))
            network_input_zero_vel_up = torch.stack(
                (self.x_up[2], torch.tensor(0, dtype=self.dtype)))
        mip_cnstr_result, _, _, _, _ = \
            self.dynamics_relu_free_pattern.output_constraint(
                network_input_lo, network_input_up,
                method=mip_utils.PropagateBoundsMethod.LP)
        mip_cnstr_result_zero_vel, _, _, _, _ = \
            self.dynamics_relu_free_pattern.output_constraint(
                network_input_zero_vel_lo, network_input_zero_vel_up,
                method=mip_utils.PropagateBoundsMethod.LP)
        # First add mip_cnstr_result. But don't impose the constraint on the
        # output of the network (we will impose the output constraint
        # afterwards inside this function)
        vel_zero_var = mip.addVars(1,
                                   lb=0,
                                   ub=0,
                                   vtype=gurobipy.GRB.CONTINUOUS,
                                   name="zero_vel")[0]
        if self.thetadot_as_input:
            input_vars = [x_var[2], u_var[0], u_var[1]]
            input_vars_zero_vel = [x_var[2], vel_zero_var, u_var[1]]
        else:
            input_vars = [x_var[2], u_var[0]]
            input_vars_zero_vel = [x_var[2], vel_zero_var]
        forward_slack, forward_binary = \
            mip.add_mixed_integer_linear_constraints(
                mip_cnstr_result, input_vars, None, slack_var_name,
                binary_var_name, "dubins_car_forward_dynamics_ineq",
                "dubins_car_forward_dyamics_eq", None)
        forward_slack_zero_vel, forward_binary_zero_vel = \
            mip.add_mixed_integer_linear_constraints(
                mip_cnstr_result_zero_vel, input_vars_zero_vel, None,
                slack_var_name + "zero_vel", binary_var_name + "zero_vel",
                "dubins_car_forward_dynamics_zero_vel_ineq",
                "dubins_car_forward_dynamics_zero_vel_eq", None)
        # Now add the constraint on the output of the network, that
        # [delta_pos_x, delta_pos_y] =
        # ϕ(θ[n], vel[n], θ_dot[n])−ϕ(θ[n], 0, θ_dot[n])
        # Namely [x_next[0] - x[0], x_next[1] - x[1]] = Aout_slack * s + Cout
        # - Aout_slack_zero_vel * s_zero_vel - Cout_zero_vel
        assert (mip_cnstr_result.Aout_input is None)
        assert (mip_cnstr_result.Aout_binary is None)
        assert (mip_cnstr_result_zero_vel.Aout_input is None)
        assert (mip_cnstr_result_zero_vel.Aout_binary is None)
        mip.addMConstrs([
            torch.eye(2, dtype=self.dtype), -torch.eye(2, dtype=self.dtype),
            -mip_cnstr_result.Aout_slack, mip_cnstr_result_zero_vel.Aout_slack
        ], [x_next_var[:2], x_var[:2], forward_slack, forward_slack_zero_vel],
                        b=mip_cnstr_result.Cout -
                        mip_cnstr_result_zero_vel.Cout,
                        sense=gurobipy.GRB.EQUAL,
                        name="dubins_car_forward_dynamics_output")

        # Now add the constraint θ[n+1] = θ[n] + θ_dot[n] * dt
        mip.addLConstr([torch.tensor([1., -1., -self.dt], dtype=self.dtype)],
                       [[x_next_var[2], x_var[2], u_var[1]]],
                       sense=gurobipy.GRB.EQUAL,
                       rhs=0.,
                       name="dubins_car_theta_dynamics")
        return forward_slack, forward_binary


class DubinsCarVisualizer:
    def __init__(self, ax, x_lim, y_lim):
        self.ax = ax
        self.ax.set_aspect("equal")
        self.ax.set_xlim(x_lim[0], x_lim[1])
        self.ax.set_ylim(y_lim[0], y_lim[1])

        self.car_length = 0.1
        self.car_width = 0.05

        self.base = np.vstack(
            (self.car_length / 2 * np.array([1, -1, -1, 1, 1]),
             self.car_width / 2 * np.array([1, 1, -1, -1, 1])))

        self.base_fill = self.ax.fill(self.base[0, :],
                                      self.base[1, :],
                                      zorder=1,
                                      edgecolor="k",
                                      facecolor=[.6, .6, .6])
        self.goal = self.ax.plot(0, 0, marker='*', markersize=10)

    def draw(self, t, x):
        theta = x[2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        p = np.dot(R, self.base)
        self.base_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.base_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]
        self.ax.set_title("t = {:.2f}s".format(t))
