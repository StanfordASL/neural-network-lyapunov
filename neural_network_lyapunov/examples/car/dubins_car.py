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
            return torch.tensor(
                [vel * torch.cos(theta), vel * torch.sin(theta), thetadot],
                dtype=self.dtype)

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
    [delta_pos_x, delta_pos_y] = ϕ(θ, vel, θ_dot) - ϕ(0, 0, 0)
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
        [pos_x[n+1], pos_y[n+1]] = [pos_x[n], pos_y[n]]
                                   + ϕ(network_input) − ϕ(0)
        θ[n+1] = θ[n] + θ̇_dot[n]*dt
        where network_input is [θ[n], vel[n], θ_dot[n]] when thetadot_as_input
        is True, and [θ[n], vel[n]] when thetadot_as_input is false.
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
                network_input_zero = torch.zeros((3, ), dtype=self.dtype)
            else:
                network_input = torch.stack((x_start[2], u_start[0]))
                network_input_zero = torch.zeros((2, ), dtype=self.dtype)
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
                network_input_zero = torch.zeros((3, ), dtype=self.dtype)
            else:
                network_input = torch.cat((x_start[:, 2].reshape(
                    (-1, 1)), u_start[:, 0].reshape((-1, 1))),
                                          dim=1)
                network_input_zero = torch.zeros((2, ), dtype=self.dtype)
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
        """
        assert (isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
        if self.thetadot_as_input:
            network_input_lo = torch.stack(
                (self.x_lo[2], self.u_lo[0], self.u_lo[1]))
            network_input_up = torch.stack(
                (self.x_up[2], self.u_up[0], self.u_up[1]))
        else:
            network_input_lo = torch.stack((self.x_lo[2], self.u_lo[0]))
            network_input_up = torch.stack((self.x_up[2], self.u_up[0]))
        mip_cnstr_result, _, _, _, _ = \
            self.dynamics_relu_free_pattern.output_constraint(
                    network_input_lo, network_input_up,
                    method=mip_utils.PropagateBoundsMethod.IA)
        # First add mip_cnstr_result. But don't impose the constraint on the
        # output of the network (we will impose the output constraint
        # afterwards inside this function)
        if self.thetadot_as_input:
            input_vars = [x_var[2], u_var[0], u_var[1]]
        else:
            input_vars = [x_var[2], u_var[0]]
        forward_slack, forward_binary = \
            mip.add_mixed_integer_linear_constraints(
                mip_cnstr_result, input_vars, None, slack_var_name,
                binary_var_name, "dubins_car_forward_dynamics_ineq",
                "dubins_car_forward_dyamics_eq", None)
        # Now add the constraint on the output of the network, that
        # [delta_pos_x, delta_pos_y] = ϕ(θ[n], vel[n], θ_dot[n])−ϕ(0, 0, 0)
        # Namely [x_next[0] - x[0], x_next[1] - x[1]] = Aout_slack * s + Cout
        # - ϕ(0, 0)
        assert (mip_cnstr_result.Aout_input is None)
        assert (mip_cnstr_result.Aout_binary is None)
        mip.addMConstrs([
            torch.eye(2, dtype=self.dtype), -torch.eye(2, dtype=self.dtype),
            -mip_cnstr_result.Aout_slack
        ], [x_next_var[:2], x_var[:2], forward_slack],
                        b=mip_cnstr_result.Cout - self.dynamics_relu(
                            torch.zeros(
                                (len(input_vars), ), dtype=self.dtype)),
                        sense=gurobipy.GRB.EQUAL,
                        name="dubins_car_forward_dynamics_output")

        # Now add the constraint θ[n+1] = θ[n] + θ_dot[n] * dt
        mip.addLConstr([torch.tensor([1., -1., -self.dt], dtype=self.dtype)],
                       [[x_next_var[2], x_var[2], u_var[1]]],
                       sense=gurobipy.GRB.EQUAL,
                       rhs=0.,
                       name="dubins_car_theta_dynamics")
        return forward_slack, forward_binary
