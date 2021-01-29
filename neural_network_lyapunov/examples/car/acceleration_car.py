"""
This car controls the linear acceleration and turning rate.
"""
import torch
import numpy as np
import scipy.integrate
import gurobipy
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.mip_utils as mip_utils


class AccelerationCar:
    """
    The state of the car is (pos_x, pos_y, theta, vel), and the control is
    (theta_dot, accel), where vel is the speed of the car along the car heading
    direction, and accel is the acceleration along the car heading direction.
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def dynamics(self, x, u):
        """
        Compute the time derivative of the state.
        """
        theta = x[2]
        vel = x[3]
        if isinstance(x, np.ndarray):
            pos_dot = np.array([np.cos(theta) * vel, np.sin(theta) * vel])
            return np.array([pos_dot[0], pos_dot[1], u[0], u[1]])
        elif isinstance(x, torch.Tensor):
            return torch.cat(
                (torch.cos(theta) * vel, torch.sin(theta) * vel, u[0], u[1]))

    def next_pose(self, x, u, dt):
        """
        Compute the next pose (x_next, y_next, yaw_next) given the current
        state (x, y, yaw, vel) and control (yaw_rate, accel) after dt.
        """
        assert (isinstance(x, np.ndarray))
        assert (isinstance(u, np.ndarray))
        result = scipy.integrate.solve_ivp(
            lambda t, x_val: self.dynamics(x_val, u), (0, dt), x)
        return result.y[:3, -1]


class AccelerationCarReLUModel:
    """
    We model the discrete-time dynamics of acceleration car using a multi-layer
    perceptron with (leaky) ReLU units.
    We consider the origin being the goal state of the car. The neural network
    predicts the next state as
    pos[n+1] - pos[n] = ϕ(theta[n], vel[n], theta_dot[n], accel[n])
                        - ϕ(0, 0, 0, 0)
    theta[n+1] = theta[n] + theta_dot[n] * dt
    vel[n+1] = vel[n] + accel[n] * dt
    where ϕ is the neural network.
    """
    def __init__(self, dtype, x_lo: torch.Tensor, x_up: torch.Tensor,
                 u_lo: torch.Tensor, u_up: torch.Tensor,
                 dynamics_relu: torch.nn.Sequential, dt: float):
        """
        @param x_lo The lower bound of the state.
        @param x_up The upper bound of the state.
        @param u_lo The lower input limits.
        @param u_up The upper input limits.
        @param dynamics_relu ϕ in the documentation above.
        @param dt The delta time (which is also used in training the dynamics
        relu).
        """
        self.dtype = dtype
        self.x_dim = 4
        assert (isinstance(x_lo, torch.Tensor))
        assert (x_lo.shape == (4, ))
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
        assert (dynamics_relu[0].in_features == 4)
        assert (dynamics_relu[-1].out_features == 2)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        self.x_equilibrium = torch.zeros((4, ), dtype=dtype)
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

    def step_forward(self, x_start: torch.Tensor,
                     u_start: torch.Tensor) -> torch.Tensor:
        """
        Compute the next state as
        pos[n+1] - pos[n] = ϕ(theta[n], vel[n], theta_dot[n], accel[n])
                            - ϕ(0, 0, 0, 0)
        theta[n+1] = theta[n] + theta_dot[n] * dt
        vel[n+1] = vel[n] + accel[n] * dt
        Note that ϕ encodes part of the discrete dynamics after dt.
        """
        assert (isinstance(x_start, torch.Tensor))
        assert (isinstance(u_start, torch.Tensor))
        if len(x_start.shape) == 1:
            theta_curr = x_start[2]
            vel_curr = x_start[3]
            theta_dot_curr = u_start[0]
            accel_curr = u_start[1]
            pos_curr = x_start[:2]
            delta_pos = self.dynamics_relu(
                torch.stack(
                    (theta_curr, vel_curr, theta_dot_curr,
                     accel_curr), )) - self.dynamics_relu(
                         torch.cat(
                             (self.x_equilibrium[2:], self.u_equilibrium),
                             dim=0))
            pos_next = pos_curr + delta_pos
            theta_next = theta_curr + theta_dot_curr * self.dt
            vel_next = vel_curr + accel_curr * self.dt
            return torch.cat((pos_next, torch.stack((theta_next, vel_next))))
        elif len(x_start.shape) == 2:
            pos_curr = x_start[:, :2]
            delta_pos = self.dynamics_relu(
                torch.cat(
                    (x_start[:, [2, 3]], u_start),
                    dim=1)) - self.dynamics_relu(
                        torch.cat(
                            (self.x_equilibrium[2:], self.u_equilibrium)))
            pos_next = pos_curr + delta_pos
            theta_vel_next = x_start[:, 2:] + u_start * self.dt
            return torch.cat((pos_next, theta_vel_next), dim=1)

    def possible_dx(self, x, u):
        """
        TODO(hongkai.dai): I will deprecate this function soon. But for now I
        keep it for backward maintanence as other systems in relu_system.py
        """
        return [self.step_forward(x, u)]

    def add_dynamics_constraint(self, mip: gurobi_torch_mip.GurobiTorchMIP,
                                x_var, x_next_var, u_var, slack_var_name,
                                binary_var_name):
        """
        Add the dynamic constraints a mixed-integer linear constraints. Refer
        to relu_system.py for the common API.
        The constraints are
        pos[n+1] - pos[n] = ϕ(theta[n], vel[n], theta_dot[n], accel[n])
                            - ϕ(0, 0, 0, 0)
        theta[n+1] = theta[n] + theta_dot[n] * dt
        vel[n+1] = vel[n] + accel[n] * dt
        """
        assert (isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
        network_input_lo = torch.stack(
            (self.x_lo[2], self.x_lo[3], self.u_lo[0], self.u_lo[1]))
        network_input_up = torch.stack(
            (self.x_up[2], self.x_up[3], self.u_up[0], self.u_up[1]))
        mip_cnstr_result, _, _, _, _, _, _ = \
            self.dynamics_relu_free_pattern.output_constraint(
                network_input_lo,
                network_input_up,
                method=mip_utils.PropagateBoundsMethod.LP)
        # First add mip_cnstr_result. But don't impose the constraint on the
        # output of the network (we will impose the output constraint
        # afterwards inside this function)
        input_vars = [x_var[2], x_var[3], u_var[0], u_var[1]]
        forward_slack, forward_binary = \
            mip.add_mixed_integer_linear_constraints(
                mip_cnstr_result, input_vars, None, slack_var_name,
                binary_var_name, "acceleration_car_forward_dynamics_ineq",
                "acceleration_car_forward_dynamics_eq", None)
        # Now add the constraint on the network output
        # pos[n+1] - pos[n] = ϕ(theta[n], vel[n], theta_dot[n], accel[n])
        #                    - ϕ(0, 0, 0, 0)
        #                    = Aout_slack * s - ϕ(0, 0, 0, 0)
        assert (mip_cnstr_result.Aout_input is None)
        assert (mip_cnstr_result.Aout_binary is None)
        mip.addMConstrs([
            torch.eye(2, dtype=self.dtype), -torch.eye(2, dtype=self.dtype),
            -mip_cnstr_result.Aout_slack
        ], [x_next_var[:2], x_var[:2], forward_slack],
                        b=mip_cnstr_result.Cout -
                        self.dynamics_relu(torch.zeros(
                            (4, ), dtype=self.dtype)),
                        sense=gurobipy.GRB.EQUAL,
                        name="acceleration_car_forward_dynamics_output")
        # Now add the constraint
        # theta[n+1] = theta[n] + theta_dot[n] * dt
        # vel[n+1] = vel[n] + accel[n] * dt
        mip.addLConstr([torch.tensor([1., -1., -self.dt], dtype=self.dtype)],
                       [[x_next_var[2], x_var[2], u_var[0]]],
                       sense=gurobipy.GRB.EQUAL,
                       rhs=0.,
                       name="acceleration_car_theta_dynamics")
        mip.addLConstr([torch.tensor([1., -1., -self.dt], dtype=self.dtype)],
                       [[x_next_var[3], x_var[3], u_var[1]]],
                       sense=gurobipy.GRB.EQUAL,
                       rhs=0.,
                       name="acceleration_car_theta_dynamics")
        return forward_slack, forward_binary
