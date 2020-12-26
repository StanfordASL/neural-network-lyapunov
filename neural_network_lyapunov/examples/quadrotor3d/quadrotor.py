import torch
import numpy as np
import gurobipy
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.geometry_transform as geometry_transform
import neural_network_lyapunov.mip_utils as mip_utils


class Quadrotor:
    """
    A quadrotor that directly commands the thrusts.
    The state is [pos_x, pos_y, pos_z, roll, pitch, yaw, pos_xdot, pos_ydot,
    pos_zdot, angular_vel_x, angular_vel_y, angular_vel_z], where
    angular_vel_x/y/z are the angular velocity measured in the body frame.
    Notice that unlike many models where uses the linear velocity in the body
    frame as the state, we use the linear velocit in the world frame as the
    state. The reason is that the update from linear velocity to next position
    is a linear constraint, and we don't need to use a neural network to encode
    this update.
    """
    def __init__(self, dtype):
        """
        The parameter of this quadrotor is obtained from
        Attitude stabilization of a VTOL quadrotor aircraft
        by Abdelhamid Tayebi and Stephen McGilvray.
        """
        self.mass = 0.468
        self.gravity = 9.81
        self.arm_length = 0.225
        # The inertia matrix is diagonal, we only store Ixx, Iyy and Izz.
        self.inertia = np.array([4.9E-3, 4.9E-3, 8.8E-3])
        # The ratio between the torque along the z axis versus the force.
        self.z_torque_to_force_factor = 1.1 / 29
        self.dtype = dtype
        self.hover_thrust = self.mass * self.gravity / 4

    def dynamics(self, x, u):
        # Compute the time derivative of the state.
        # The dynamics is explained in
        # Minimum Snap Trajectory Generation and Control for Quadrotors
        # by Daniel Mellinger and Vijay Kumar
        # @param u the thrust generated in each propeller
        rpy = x[3:6]
        pos_dot = x[6:9]
        omega = x[9:12]

        if isinstance(x, np.ndarray):
            # plant_input is [total_thrust, torque_x, torque_y, torque_z]
            plant_input = np.array([[1, 1, 1, 1],
                                    [0, self.arm_length, 0, -self.arm_length],
                                    [-self.arm_length, 0, self.arm_length, 0],
                                    [
                                        self.z_torque_to_force_factor,
                                        -self.z_torque_to_force_factor,
                                        self.z_torque_to_force_factor,
                                        -self.z_torque_to_force_factor
                                    ]]) @ u
            R = geometry_transform.rpy2rotmat(rpy)
            pos_ddot = np.array([
                0, 0, -self.gravity
            ]) + R @ np.array([0, 0, plant_input[0]]) / self.mass
            # Here we exploit the fact that the inertia matrix is diagonal.
            omega_dot = (np.cross(-omega, self.inertia * omega) +
                         plant_input[1:]) / self.inertia
            # Convert the angular velocity to the roll-pitch-yaw time
            # derivative.
            sin_roll = np.sin(rpy[0])
            cos_roll = np.cos(rpy[0])
            tan_pitch = np.tan(rpy[1])
            cos_pitch = np.cos(rpy[1])
            # Equation 2.7 in quadrotor control: modeling, nonlinear control
            # design and simulation by Francesco Sabatino
            rpy_dot = np.array(
                [[1., sin_roll * tan_pitch, cos_roll * tan_pitch],
                 [0., cos_roll, -sin_roll],
                 [0, sin_roll / cos_pitch, cos_roll / cos_pitch]]) @ omega
            return np.hstack((pos_dot, rpy_dot, pos_ddot, omega_dot))
        elif isinstance(x, torch.Tensor):
            # plant_input is [total_thrust, torque_x, torque_y, torque_z]
            plant_input = torch.tensor(
                [[1, 1, 1, 1], [0, self.arm_length, 0, -self.arm_length],
                 [-self.arm_length, 0, self.arm_length, 0],
                 [
                     self.z_torque_to_force_factor,
                     -self.z_torque_to_force_factor,
                     self.z_torque_to_force_factor,
                     -self.z_torque_to_force_factor
                 ]],
                dtype=x.dtype) @ u
            R = geometry_transform.rpy2rotmat(rpy)
            pos_ddot = torch.tensor(
                [0, 0, -self.gravity],
                dtype=x.dtype) + R[:, 2] * plant_input[0] / self.mass
            # Here we exploit the fact that the inertia matrix is diagonal.
            omega_dot = (torch.cross(-omega,
                                     torch.from_numpy(self.inertia) * omega) +
                         plant_input[1:]) / torch.from_numpy(self.inertia)
            # Convert the angular velocity to the roll-pitch-yaw time
            # derivative.
            sin_roll = torch.sin(rpy[0])
            cos_roll = torch.cos(rpy[0])
            tan_pitch = torch.tan(rpy[1])
            cos_pitch = torch.cos(rpy[1])
            # Equation 2.7 in quadrotor control: modeling, nonlinear control
            # design and simulation by Francesco Sabatino
            T = torch.zeros((3, 3), dtype=x.dtype)
            T[0, 0] = 1
            T[0, 1] = sin_roll * tan_pitch
            T[0, 2] = cos_roll * tan_pitch
            T[1, 1] = cos_roll
            T[1, 2] = -sin_roll
            T[2, 1] = sin_roll / cos_pitch
            T[2, 2] = cos_roll / cos_pitch
            rpy_dot = T @ omega
            return torch.cat((pos_dot, rpy_dot, pos_ddot, omega_dot))


class QuadrotorWithPixhawkReLUSystem:
    """
    This system models the high level dynamics of a quadrotor with pixhawk low
    level controller. The system state is
    state = [xyz position, roll-pitch-yaw, xyz velocity]
    The system action is
    action = [attitude_setpoint, thrust]
    The system dynamics is represented by a neural network, which takes the
    input as the current x/y/z velocity together with the orientation
    (roll-pitch-yaw), plus the command of the attitude setpoint
    and total thrust. The neural network is used to predict the delta x/y/z
    velocity, together with the change of roll-pitch-yaw.
    Namely the dynamics is
    [delta_vel, delta_rpy] = ϕ(vel, rpy, attitude_setpoint, thrust)
                             - ϕ(0, 0, 0, hover_thrust)
    where ϕ is the neural network. Notice that ϕ doesn't output the next state
    directly.
    """
    def __init__(self, dtype, x_lo: torch.Tensor, x_up: torch.Tensor,
                 u_lo: torch.Tensor, u_up: torch.Tensor,
                 dynamics_relu: torch.nn.Sequential, hover_thrust: float,
                 dt: float):
        """
        @param dynamics_relu Then network ϕ in the documentation above.
        @param hover_thrust The thrust command to maintain equilibrium.
        @param dt The delta time between two consecutive state measurement.
        """
        self.dtype = dtype
        self.x_dim = 9
        assert (isinstance(x_lo, torch.Tensor))
        assert (x_lo.shape == (self.x_dim, ))
        self.x_lo = x_lo
        assert (isinstance(x_up, torch.Tensor))
        assert (x_up.shape == (self.x_dim, ))
        self.x_up = x_up
        self.u_dim = 4
        assert (isinstance(u_lo, torch.Tensor))
        assert (u_lo.shape == (self.u_dim, ))
        self.u_lo = u_lo
        assert (isinstance(u_up, torch.Tensor))
        assert (u_up.shape == (self.u_dim, ))
        self.u_up = u_up

        assert (dynamics_relu[0].in_features == 10)
        assert (dynamics_relu[-1].out_features == 6)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        self.x_equilibrium = torch.zeros((self.x_dim, ), dtype=self.dtype)
        self.u_equilibrium = torch.tensor([0, 0, 0, hover_thrust],
                                          dtype=self.dtype)
        assert (isinstance(dt, float))
        assert (dt > 0)
        self.dt = dt

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def step_forward(self, x_start: torch.Tensor, u_start: torch.Tensor):
        """
        Compute the next state according to
        pos[n+1] = pos[n] + (v[n] + v[n+1]) * dt / 2
        [vel[n+1], rpy[n+1]] = [vel[n], rpy[n]] + ϕ(vel[n], rpy[n], u[n])
                               - ϕ(0, 0, u*)
        """
        assert (isinstance(x_start, torch.Tensor))
        assert (isinstance(u_start, torch.Tensor))
        if len(x_start.shape) == 1:
            # A single state.
            pos_current = x_start[:3]
            rpy_current = x_start[3:6]
            vel_current = x_start[6:9]
            delta_vel_rpy = self.dynamics_relu(
                torch.cat(
                    (vel_current, rpy_current, u_start),
                    dim=0)) - self.dynamics_relu(
                        torch.cat((torch.zeros(
                            (6, ), dtype=self.dtype), self.u_equilibrium),
                                  dim=0))
            vel_next = vel_current + delta_vel_rpy[:3]
            rpy_next = rpy_current + delta_vel_rpy[3:]
            pos_next = pos_current + (vel_next + vel_current) / 2 * self.dt
            return torch.cat((pos_next, rpy_next, vel_next), dim=0)
        elif len(x_start.shape) == 2:
            # A batch of states/controls.
            assert (x_start.shape[1] == self.x_dim)
            assert (u_start.shape[1] == self.u_dim)
            pos_current = x_start[:, :3]
            rpy_current = x_start[:, 3:6]
            vel_current = x_start[:, 6:9]
            delta_vel_rpy = self.dynamics_relu(
                torch.cat(
                    (vel_current, rpy_current, u_start),
                    dim=1)) - self.dynamics_relu(
                        torch.cat((torch.zeros(
                            (6, ), dtype=self.dtype), self.u_equilibrium),
                                  dim=0))
            vel_next = vel_current + delta_vel_rpy[:, :3]
            rpy_next = rpy_current + delta_vel_rpy[:, 3:6]
            pos_next = pos_current + (vel_next + vel_current) / 2 * self.dt
            return torch.cat((pos_next, rpy_next, vel_next), dim=1)

    def possible_dx(self, x, u):
        return [self.step_forward(x, u)]

    def add_dynamics_constraint(self, mip, x_var, x_next_var, u_var,
                                slack_var_name, binary_var_name):
        """
        Add the dynamic constraint
        pos[n+1] = pos[n] + (v[n] + v[n+1]) * dt / 2
        [vel[n+1], rpy[n+1]] = [vel[n], rpy[n]] + ϕ(vel[n], rpy[n], u[n])
                               - ϕ(0, 0, u*)
        as mixed-integer linear constraints.
        """
        mip_cnstr_result, _, _, _, _ = self.dynamics_relu_free_pattern.\
            output_constraint(
                    torch.cat((self.x_lo[6:9], self.x_lo[3:6], self.u_lo)),
                    torch.cat((self.x_up[6:9], self.x_up[3:6], self.u_up)),
                    mip_utils.PropagateBoundsMethod.IA)
        # First add mip_cnstr_result, but don't impose the constraint on the
        # output of the network (we will impose the constraint separately)
        vel_curr = x_var[6:9]
        rpy_curr = x_var[3:6]
        input_vars = vel_curr + rpy_curr + u_var
        forward_slack, forward_binary = \
            mip.add_mixed_integer_linear_constraints(
                mip_cnstr_result, input_vars, None, slack_var_name,
                binary_var_name, "quadrotor_forward_dynamics_ineq",
                "quadrotor_forward_dynamics_eq", None)
        # Impose the constraint
        # [vel[n+1], rpy[n+1]] = [vel[n], rpy[n]] +ϕ(vel[n], rpy[n], u[n])
        #                       - ϕ(0, 0, u*)
        #                      = [vel[n], rpy[n]] + Aout_slack * s + Cout
        #                        - ϕ(0, 0, u*)
        assert (mip_cnstr_result.Aout_input is None)
        assert (mip_cnstr_result.Aout_binary is None)
        vel_next = x_next_var[6:9]
        rpy_next = x_next_var[3:6]
        vel_rpy_next = vel_next + rpy_next
        vel_rpy_curr = vel_curr + rpy_curr
        mip.addMConstrs(
            [
                torch.eye(6, dtype=self.dtype),
                -torch.eye(6, dtype=self.dtype), -mip_cnstr_result.Aout_slack
            ], [vel_rpy_next, vel_rpy_curr, forward_slack],
            b=mip_cnstr_result.Cout - self.dynamics_relu(
                torch.cat((torch.zeros(
                    (6, ), dtype=self.dtype), self.u_equilibrium))),
            sense=gurobipy.GRB.EQUAL,
            name="quadrotor_forward_dynamics_output")
        # Now add the constraint pos[n+1] = pos[n] + (v[n] + v[n+1]) * dt / 2
        pos_next = x_next_var[:3]
        pos_curr = x_var[:3]
        mip.addMConstrs([
            torch.eye(3, dtype=self.dtype), -torch.eye(3, dtype=self.dtype),
            -self.dt / 2 * torch.eye(3, dtype=self.dtype),
            -self.dt / 2 * torch.eye(3, dtype=self.dtype)
        ], [pos_next, pos_curr, vel_next, vel_curr],
                        b=torch.zeros((3, ), dtype=self.dtype),
                        sense=gurobipy.GRB.EQUAL,
                        name="update_pos_next")
        return forward_slack, forward_binary
