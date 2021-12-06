import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_system as relu_system

import torch
import gurobipy


class PoleReluSystem:
    """
    Model the pole system using a relu network
    x_AB[n+1] = x_AB[n] + (ẋ_AB[n] + ẋ_AB[n+1]) * dt / 2
    y_AB[n+1] = y_AB[n] + (ẏ_AB[n] + ẏ_AB[n+1]) * dt / 2
    [ẋ_A[n+1] ẏ_A[n+1] ż_A[n+1] ẋ_AB[n+1] ẏ_AB[n+1]] =
    [ẋ_A[n] ẏ_A[n] ż_A[n] ẋ_AB[n] ẏ_AB[n]] +
    ϕ(x_AB[n], y_AB[n], ẋ_AB[n], ẏ_AB[n], u[n]) -
    ϕ(x_AB*, y_AB*, ẋ_AB*, ẏ_AB*, u*)
    where ϕ is the neural network.
    """
    def __init__(self, x_lo: torch.Tensor, x_up: torch.Tensor,
                 u_lo: torch.Tensor, u_up: torch.Tensor, dynamics_relu,
                 dt: float, u_z_equilibrium: float):
        """
        Args:
          u_z_equilibrium: The equilibrium input force in the z direction
          (which is the total gravitational force of the pole/end-effector
          system).
        """
        assert (isinstance(x_lo, torch.Tensor))
        assert (isinstance(x_up, torch.Tensor))
        assert (x_lo.shape == (7, ))
        assert (x_up.shape == (7, ))
        self.dtype = x_lo.dtype
        self.x_dim = 7
        self.u_dim = 3
        self.x_lo = x_lo
        self.x_up = x_up
        assert (isinstance(u_lo, torch.Tensor))
        assert (isinstance(u_up, torch.Tensor))
        self.u_lo = u_lo
        self.u_up = u_up
        self.dt = dt
        self.x_equilibrium = torch.zeros((self.x_dim, ), dtype=self.dtype)
        self.u_equilibrium = torch.tensor([0, 0, u_z_equilibrium],
                                          dtype=self.dtype)
        assert (dynamics_relu[0].in_features == 7)
        assert (dynamics_relu[-1].out_features == 5)
        self.dynamics_relu = dynamics_relu
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, self.dtype)
        self.network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.IA

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def step_forward(self, x_start: torch.Tensor, u_start: torch.Tensor):
        if len(x_start.shape) == 1:
            v_next = x_start[2:] + self.dynamics_relu(
                torch.cat(
                    (x_start[:2], x_start[5:], u_start))) - self.dynamics_relu(
                        torch.cat(
                            (self.x_equilibrium[:2], self.x_equilibrium[5:],
                             self.u_equilibrium)))
            xy_AB_next = x_start[:2] + (v_next[-2:] +
                                        x_start[-2:]) * self.dt / 2
            return torch.cat((xy_AB_next, v_next))
        elif len(x_start.shape) == 2:
            v_next = x_start[:, 2:] + self.dynamics_relu(
                torch.cat(
                    (x_start[:, :2], x_start[:, 5:], u_start),
                    dim=1)) - self.dynamics_relu(
                        torch.cat(
                            (self.x_equilibrium[:2], self.x_equilibrium[5:],
                             self.u_equilibrium)))
            xy_AB_next = x_start[:, :2] + (v_next[:, -2:] +
                                           x_start[:, -2:]) * self.dt / 2
            return torch.cat((xy_AB_next, v_next), dim=1)

    def possible_dx(self, x, u):
        return [self.step_forward(x, u)]

    def add_dynamics_constraint(self,
                                mip: gurobi_torch_mip.GurobiTorchMIP,
                                x_var,
                                x_next_var,
                                u_var,
                                additional_u_lo: torch.Tensor = None,
                                additional_u_up: torch.Tensor = None,
                                binary_var_type=gurobipy.GRB.BINARY):
        u_lo = self.u_lo if additional_u_lo is None else torch.max(
            self.u_lo, additional_u_lo)
        u_up = self.u_up if additional_u_up is None else torch.min(
            self.u_up, additional_u_up)
        mip_cnstr_result = self.dynamics_relu_free_pattern.output_constraint(
            torch.cat((self.x_lo[:2], self.x_lo[-2:], u_lo)),
            torch.cat((self.x_up[:2], self.x_up[-2:], u_up)),
            self.network_bound_propagate_method)
        # First add mip_cnstr_result to mip, but don't impose the constraint
        # on the output of the network (we will impose the output constraint
        # separately).
        forward_slack, forward_binary = \
            mip.add_mixed_integer_linear_constraints(
                mip_cnstr_result, x_var[:2] + x_var[-2:] + u_var, None,
                "pole_dynamics_slack", "pole_dynamics_binary",
                "pole_dynamics_ineq", "pole_dynamics_eq", None,
                binary_var_type)
        # Impose the constraint
        # [ẋ_A[n+1] ẏ_A[n+1] ż_A[n+1] ẋ_AB[n+1] ẏ_AB[n+1]] =
        # [ẋ_A[n] ẏ_A[n] ż_A[n] ẋ_AB[n] ẏ_AB[n]] +
        # ϕ(x_AB[n], y_AB[n], ẋ_AB[n], ẏ_AB[n], u[n]) -
        # ϕ(x_AB*, y_AB*, ẋ_AB*, ẏ_AB*, u*)
        assert (mip_cnstr_result.Aout_input is None)
        assert (mip_cnstr_result.Aout_binary is None)
        mip.addMConstrs(
            [
                torch.eye(5,
                          dtype=self.dtype), -torch.eye(5, dtype=self.dtype),
                -mip_cnstr_result.Aout_slack.reshape(
                    (mip_cnstr_result.num_out(), mip_cnstr_result.num_slack()))
            ], [x_next_var[2:], x_var[2:], forward_slack],
            b=mip_cnstr_result.Cout.reshape(
                (mip_cnstr_result.num_out(), )) - self.dynamics_relu(
                    torch.cat((self.x_equilibrium[:2], self.x_equilibrium[-2:],
                               self.u_equilibrium))),
            sense=gurobipy.GRB.EQUAL,
            name="pole_dynamics_output")
        # Now add the constraint
        # x_AB[n+1] = x_AB[n] + (ẋ_AB[n] + ẋ_AB[n+1]) * dt / 2
        # y_AB[n+1] = y_AB[n] + (ẏ_AB[n] + ẏ_AB[n+1]) * dt / 2
        mip.addMConstrs([
            torch.eye(2, dtype=self.dtype), -torch.eye(2, dtype=self.dtype),
            -self.dt / 2 * torch.eye(2, dtype=self.dtype),
            -self.dt / 2 * torch.eye(2, dtype=self.dtype)
        ], [x_next_var[:2], x_var[:2], x_next_var[-2:], x_var[-2:]],
                        b=torch.tensor([0, 0], dtype=self.dtype),
                        sense=gurobipy.GRB.EQUAL)
        ret = relu_system.ReLUDynamicsConstraintReturn(forward_slack,
                                                       forward_binary)
        nn_input_vars = x_var[:2] + x_var[-2:] + u_var
        ret.from_mip_cnstr_return(mip_cnstr_result, nn_input_vars)
        return ret
