import torch
import gurobipy
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.mip_utils as mip_utils


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
