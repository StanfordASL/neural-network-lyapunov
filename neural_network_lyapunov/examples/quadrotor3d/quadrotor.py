import torch
import numpy as np
import scipy
import gurobipy
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.relu_system as relu_system
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

    def dynamics_gradient(self, x: np.ndarray, u: np.ndarray):
        """
        Compute the dynamics gradient A = ∂f/∂x, B = ∂f/∂u
        """
        assert (isinstance(x, np.ndarray))
        assert (isinstance(u, np.ndarray))
        rpy = x[3:6]
        omega = x[9:12]
        A = np.zeros((12, 12))
        B = np.zeros((12, 4))
        # ∂position_dot / ∂vel
        A[0:3, 6:9] = np.eye(3)
        sin_roll = np.sin(rpy[0])
        cos_roll = np.cos(rpy[0])
        tan_pitch = np.tan(rpy[1])
        cos_pitch = np.cos(rpy[1])
        sec_pitch = 1 / cos_pitch
        drpy_dot_droll = np.array(
            [[0, cos_roll * tan_pitch, -sin_roll * tan_pitch],
             [0, -sin_roll, -cos_roll],
             [0, cos_roll / cos_pitch, -sin_roll / cos_pitch]]) @ omega
        drpy_dot_dpitch = np.array(
            [[0, sin_roll * (sec_pitch**2), cos_roll *
              (sec_pitch**2)], [0, 0, 0],
             [
                 0, sin_roll * tan_pitch * sec_pitch,
                 cos_roll * tan_pitch * sec_pitch
             ]]) @ omega
        A[3:6, 3] = drpy_dot_droll
        A[3:6, 4] = drpy_dot_dpitch
        drpy_dot_domega = np.array(
            [[1., sin_roll * tan_pitch, cos_roll * tan_pitch],
             [0., cos_roll, -sin_roll],
             [0, sin_roll / cos_pitch, cos_roll / cos_pitch]])
        A[3:6, 9:12] = drpy_dot_domega
        dR_droll, dR_dpitch, dR_dyaw = geometry_transform.rpy2rotmat_gradient(
            rpy)
        dpos_ddot_droll = dR_droll @ np.array([0, 0, u.sum()]) / self.mass
        dpos_ddot_dpitch = dR_dpitch @ np.array([0, 0, u.sum()]) / self.mass
        dpos_ddot_dyaw = dR_dyaw @ np.array([0, 0, u.sum()]) / self.mass
        A[6:9, 3] = dpos_ddot_droll
        A[6:9, 4] = dpos_ddot_dpitch
        A[6:9, 5] = dpos_ddot_dyaw
        R = geometry_transform.rpy2rotmat(rpy)
        for i in range(4):
            B[6:9, i] = R @ np.array([0, 0, 1]) / self.mass

        domegadot_domega = np.array([
            [
                0, (self.inertia[1] - self.inertia[2]) / self.inertia[0] *
                omega[2], (self.inertia[1] - self.inertia[2]) /
                self.inertia[0] * omega[1]
            ],
            [(self.inertia[2] - self.inertia[0]) / self.inertia[1] * omega[2],
             0,
             (self.inertia[2] - self.inertia[0]) / self.inertia[1] * omega[0]],
            [(self.inertia[0] - self.inertia[1]) / self.inertia[2] * omega[1],
             (self.inertia[0] - self.inertia[1]) / self.inertia[2] * omega[0],
             0]
        ])
        A[9:12, 9:12] = domegadot_domega
        B[9, :] = np.array([0, self.arm_length, 0, -self.arm_length
                            ]) / self.inertia[0]
        B[10, :] = np.array([-self.arm_length, 0, self.arm_length, 0
                             ]) / self.inertia[1]
        B[11, :] = np.array([
            self.z_torque_to_force_factor, -self.z_torque_to_force_factor,
            self.z_torque_to_force_factor, -self.z_torque_to_force_factor
        ]) / self.inertia[2]
        return A, B

    def lqr_control(self, Q, R, x, u):
        x_np = x if isinstance(x, np.ndarray) else x.detach().numpy()
        u_np = x if isinstance(u, np.ndarray) else u.detach().numpy()
        A, B = self.dynamics_gradient(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K, S


class QuadrotorWithPixhawkReLUSystem:
    """
    This system models the high level dynamics of a quadrotor with pixhawk low
    level controller. The system state is
    state = [xyz position, roll-pitch-yaw, xyz velocity]
    The system action is
    action = [attitude_setpoint, thrust]
    The system dynamics is represented by a neural network, which takes the
    input as the current orientation (roll-pitch-yaw), plus the command of the
    attitude setpoint and total thrust. The neural network is used to predict
    the delta x/y/z velocity, together with the change of roll-pitch-yaw.
    Namely the dynamics is
    [delta_vel, delta_rpy] = ϕ(rpy, attitude_setpoint, thrust)
                             - ϕ(0, 0, hover_thrust)
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

        assert (dynamics_relu[0].in_features == 7)
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
        [vel[n+1], rpy[n+1]] = [vel[n], rpy[n]] + ϕ(rpy[n], u[n])
                               - ϕ(0, u*)
        """
        assert (isinstance(x_start, torch.Tensor))
        assert (isinstance(u_start, torch.Tensor))
        if len(x_start.shape) == 1:
            # A single state.
            pos_current = x_start[:3]
            rpy_current = x_start[3:6]
            vel_current = x_start[6:9]
            delta_vel_rpy = self.dynamics_relu(
                torch.cat((rpy_current, u_start), dim=0)) - self.dynamics_relu(
                    torch.cat((torch.zeros(
                        (3, ), dtype=self.dtype), self.u_equilibrium),
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
                torch.cat((rpy_current, u_start), dim=1)) - self.dynamics_relu(
                    torch.cat((torch.zeros(
                        (3, ), dtype=self.dtype), self.u_equilibrium),
                              dim=0))
            vel_next = vel_current + delta_vel_rpy[:, :3]
            rpy_next = rpy_current + delta_vel_rpy[:, 3:6]
            pos_next = pos_current + (vel_next + vel_current) / 2 * self.dt
            return torch.cat((pos_next, rpy_next, vel_next), dim=1)

    def possible_dx(self, x, u):
        return [self.step_forward(x, u)]

    def add_dynamics_constraint(self,
                                mip,
                                x_var,
                                x_next_var,
                                u_var,
                                slack_var_name,
                                binary_var_name,
                                additional_u_lo: torch.Tensor = None,
                                additional_u_up: torch.Tensor = None,
                                binary_var_type=gurobipy.GRB.BINARY):
        """
        Add the dynamic constraint
        pos[n+1] = pos[n] + (v[n] + v[n+1]) * dt / 2
        [vel[n+1], rpy[n+1]] = [vel[n], rpy[n]] + ϕ(rpy[n], u[n])
                               - ϕ(0, u*)
        as mixed-integer linear constraints.
        """
        u_lo = self.u_lo if additional_u_lo is None else torch.max(
            self.u_lo, additional_u_lo)
        u_up = self.u_up if additional_u_up is None else torch.min(
            self.u_up, additional_u_up)
        mip_cnstr_result = self.dynamics_relu_free_pattern.output_constraint(
            torch.cat((self.x_lo[3:6], u_lo)),
            torch.cat((self.x_up[3:6], u_up)),
            mip_utils.PropagateBoundsMethod.IA)
        # First add mip_cnstr_result, but don't impose the constraint on the
        # output of the network (we will impose the constraint separately)
        vel_curr = x_var[6:9]
        rpy_curr = x_var[3:6]
        input_vars = rpy_curr + u_var
        forward_slack, forward_binary = \
            mip.add_mixed_integer_linear_constraints(
                mip_cnstr_result, input_vars, None, slack_var_name,
                binary_var_name, "quadrotor_forward_dynamics_ineq",
                "quadrotor_forward_dynamics_eq", None, binary_var_type)
        # Impose the constraint
        # [vel[n+1], rpy[n+1]] = [vel[n], rpy[n]] + ϕ(rpy[n], u[n])
        #                       - ϕ(0, u*)
        #                      = [vel[n], rpy[n]] + Aout_slack * s + Cout
        #                        - ϕ(0, u*)
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
                    (3, ), dtype=self.dtype), self.u_equilibrium))),
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
        ret = relu_system.ReLUDynamicsConstraintReturn(
            forward_slack, forward_binary)
        ret.from_mip_cnstr_return(mip_cnstr_result, input_vars)
        return ret


class QuadrotorReLUSystem:
    """
    Approximate the discrete-time dynamics of a quadrotor (with thrust as the
    input) using a neural network (with ReLU activations).
    The neural network takes the input as (rpy[n], angular_vel[n], thrust[n]).
    The dynamics is
    pos[n+1] = pos[n] + (pos_dot[n] + pos_dot[n+1]) / 2 * dt
    (rpy[n+1], pos_dot[n+1] - pos_dot[n], angular_vel[n+1])
    = ϕ(rpy[n], angular_vel[n], thrust[n]) - ϕ(0, 0, hover_thrust)
    where ϕ is the neural network. Note here we use the fact that the quadrotor
    acceleration doesn't depend on its position and linear velocity.
    """
    def __init__(self, dtype, x_lo: torch.Tensor, x_up: torch.Tensor,
                 u_lo: torch.Tensor, u_up: torch.Tensor,
                 dynamics_relu: torch.nn.Sequential, hover_thrust: float,
                 dt: float):
        """
        @param u_lo The lower limit of the input. Note that the system input is
        the thrust, so this lower limit should be non-negative.
        @param dynamics_relu The neural network ϕ in the documentation above.
        @param hover_thrust a float. The hovering thrust for each motor.
        """
        self.dtype = dtype
        self.x_dim = 12
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
        assert (dynamics_relu[-1].out_features == 9)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        self.x_equilibrium = torch.zeros((self.x_dim, ), dtype=self.dtype)
        self.u_equilibrium = hover_thrust * torch.ones((4, ), dtype=self.dtype)
        assert (isinstance(dt, float))
        assert (dt > 0)
        self.dt = dt
        self.network_bound_propagate_method =\
            mip_utils.PropagateBoundsMethod.IA

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def step_forward(self, x_start: torch.Tensor, u_start: torch.Tensor):
        """
        Compute the next state as
        pos[n+1] = pos[n] + (pos_dot[n] + pos_dot[n+1]) / 2 * dt
        (rpy[n+1], pos_dot[n+1]-pos_dot[n], angular_vel[n+1])
        = ϕ(rpy[n], angular_vel[n], thrust[n]) - ϕ(0, 0, hover_thrust)
        """
        assert (isinstance(x_start, torch.Tensor))
        assert (isinstance(u_start, torch.Tensor))
        if len(x_start.shape) == 1:
            # single state.
            rpy_delta_posdot_angularvel_next = self.dynamics_relu(
                torch.cat(
                    (x_start[3:6], x_start[9:12], u_start),
                    dim=0)) - self.dynamics_relu(
                        torch.cat(
                            (self.x_equilibrium[3:6], self.x_equilibrium[9:12],
                             self.u_equilibrium),
                            dim=0))
            posdot_next = x_start[6:9] + rpy_delta_posdot_angularvel_next[3:6]
            pos_next = x_start[:3] + (x_start[6:9] + posdot_next) / 2 * self.dt
            rpy_next = rpy_delta_posdot_angularvel_next[:3]
            angularvel_next = rpy_delta_posdot_angularvel_next[6:9]
            return torch.cat(
                (pos_next, rpy_next, posdot_next, angularvel_next), dim=0)
        elif len(x_start.shape) == 2:
            # batch of state/control.
            rpy_delta_posdot_angularvel_next = self.dynamics_relu(
                torch.cat(
                    (x_start[:, 3:6], x_start[:, 9:12], u_start),
                    dim=1)) - self.dynamics_relu(
                        torch.cat(
                            (self.x_equilibrium[3:6], self.x_equilibrium[9:12],
                             self.u_equilibrium),
                            dim=0))
            posdot_next = x_start[:,
                                  6:9] + rpy_delta_posdot_angularvel_next[:,
                                                                          3:6]
            pos_next = x_start[:, :3] + (x_start[:, 6:9] +
                                         posdot_next) / 2 * self.dt
            rpy_next = rpy_delta_posdot_angularvel_next[:, :3]
            angularvel_next = rpy_delta_posdot_angularvel_next[:, 6:9]
            return torch.cat(
                (pos_next, rpy_next, posdot_next, angularvel_next), dim=-1)

    def possible_dx(self, x, u):
        return [self.step_forward(x, u)]

    def _add_dynamics_constraint_given_relu_bounds(
            self, mip, x_var, x_next_var, u_var, slack_var_name,
            binary_var_name, relu_input_lo, relu_input_up, relu_output_lo,
            relu_output_up, network_input_lo,
            network_input_up, binary_var_type):
        mip_cnstr_result = self.dynamics_relu_free_pattern._output_constraint_given_bounds(  # noqa
            relu_input_lo, relu_input_up, network_input_lo, network_input_up)
        # First add mip_cnstr_result, but don't impose the constraint on the
        # output of the network (we will impose the constraint separately)
        input_vars = x_var[3:6] + x_var[9:12] + u_var
        forward_slack, forward_binary = \
            mip.add_mixed_integer_linear_constraints(
                mip_cnstr_result, input_vars, None, slack_var_name,
                binary_var_name, "quadrotor_dynamics_ineq",
                "quadrotor_dynamics_eq", None, binary_var_type)
        # Impose the constraint
        # (rpy[n+1], pos_dot[n+1]-pos_dot[n], angular_vel[n+1])
        # = ϕ(rpy[n], angular_vel[n], thrust[n]) - ϕ(0, 0, hover_thrust)
        # = Aout_slack * s + Cout - ϕ(0, 0, hover_thrust)
        assert (mip_cnstr_result.Aout_input is None)
        assert (mip_cnstr_result.Aout_binary is None)
        posdot_next = x_next_var[6:9]
        posdot_curr = x_var[6:9]
        posdot_curr_coeff = torch.zeros((9, 3), dtype=self.dtype)
        posdot_curr_coeff[3:6, :] = -torch.eye(3, dtype=self.dtype)
        mip.addMConstrs(
            [
                torch.eye(9, dtype=self.dtype), posdot_curr_coeff,
                -mip_cnstr_result.Aout_slack
            ], [x_next_var[3:12], posdot_curr, forward_slack],
            b=mip_cnstr_result.Cout - self.dynamics_relu(
                torch.cat((self.x_equilibrium[3:6], self.x_equilibrium[9:12],
                           self.u_equilibrium))),
            sense=gurobipy.GRB.EQUAL,
            name="quadrotor_dynamics_output")
        # Now add the constraint
        # pos[n+1] - pos[n] = (posdot[n+1] + posdot[n]) * dt / 2
        pos_next = x_next_var[:3]
        pos_curr = x_var[:3]
        mip.addMConstrs([
            torch.eye(3, dtype=self.dtype), -torch.eye(3, dtype=self.dtype),
            -self.dt / 2 * torch.eye(3, dtype=self.dtype),
            -self.dt / 2 * torch.eye(3, dtype=self.dtype)
        ], [pos_next, pos_curr, posdot_next, posdot_curr],
                        b=torch.zeros((3, ), dtype=self.dtype),
                        sense=gurobipy.GRB.EQUAL,
                        name="update_pos")
        ret = relu_system.ReLUDynamicsConstraintReturn(
            forward_slack, forward_binary)
        ret.from_mip_cnstr_return(mip_cnstr_result, input_vars)
        ret.relu_output_lo = relu_output_lo
        ret.relu_output_up = relu_output_up
        return ret

    def add_dynamics_constraint(self,
                                mip,
                                x_var,
                                x_next_var,
                                u_var,
                                slack_var_name,
                                binary_var_name,
                                additional_u_lo: torch.Tensor = None,
                                additional_u_up: torch.Tensor = None,
                                create_lp_prog_callback=None,
                                binary_var_type=gurobipy.GRB.BINARY):
        """
        Add the dynamics constraints
        pos[n+1] = pos[n] + (pos_dot[n] + pos_dot[n+1]) / 2 * dt
        (rpy[n+1], pos_dot[n+1] - pos_dot[n], angular_vel[n+1])
        = ϕ(rpy[n],, angular_vel[n], thrust[n]) - ϕ(0, 0, hover_thrust)
        @param additional_u_lo The additional lower bound on u.
        @param additional_u_up The additional upper bound on u.
        @param create_lp_prog_callback Only used when propagating the bounds of
        the network ReLU inputs using LP. This callback will be used to add
        additional constraints to the LP. The additional constraints include
        those from the feedback system when connecting this forward system with
        a controller.
        """
        u_lo = self.u_lo if additional_u_lo is None else torch.max(
            self.u_lo, additional_u_lo)
        u_up = self.u_up if additional_u_up is None else torch.min(
            self.u_up, additional_u_up)

        network_input_lo = torch.cat((self.x_lo[3:6], self.x_lo[9:12], u_lo))
        network_input_up = torch.cat((self.x_up[3:6], self.x_up[9:12], u_up))

        relu_input_lo, relu_input_up, relu_output_lo, relu_output_up =\
            self.dynamics_relu_free_pattern._compute_layer_bound(
                network_input_lo, network_input_up,
                self.network_bound_propagate_method,
                create_prog_callback=create_lp_prog_callback)

        return self._add_dynamics_constraint_given_relu_bounds(
            mip, x_var, x_next_var, u_var, slack_var_name, binary_var_name,
            relu_input_lo, relu_input_up, relu_output_lo, relu_output_up,
            network_input_lo, network_input_up, binary_var_type)
