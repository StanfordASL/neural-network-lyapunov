import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.utils as utils

import torch
import numpy as np
import gurobipy


class UnicycleFeedbackSystem(feedback_system.FeedbackSystem):
    """
    Derived from FeedbackSystem, but the controller has the form
    [vel, theta_dot] = clamp(ϕ(x) − ϕ(0) + [λᵤ |Rᵤ*[p_x, p_y]|₁, 0],
             [0, thetadot_min)], [vel_max, thetadot_max])
    This controller guarantees that locally around the origin (p_x = p_y = 0),
    the output velocity is strictly positive.
    """
    def __init__(self, forward_system, controller_network,
                 u_lower_limit: np.ndarray, u_upper_limit: np.ndarray,
                 lambda_u: float, Ru_options: r_options.ROptions):
        super(UnicycleFeedbackSystem,
              self).__init__(forward_system, controller_network,
                             torch.zeros((3, ), dtype=torch.float64),
                             torch.zeros((2, ), dtype=torch.float64),
                             u_lower_limit, u_upper_limit)
        assert (u_lower_limit[0] == 0)
        assert (isinstance(lambda_u, float))
        assert (lambda_u > 0)
        self.lambda_u = lambda_u
        assert (isinstance(Ru_options, r_options.ROptions))
        self.Ru_options = Ru_options

    def compute_u(self, x):
        assert (isinstance(x, torch.Tensor))
        if x.shape == (3, ):
            u_pre_sat = self.controller_network(x) - self.controller_network(
                torch.zeros((3, ), dtype=self.dtype)) + torch.stack(
                    (self.lambda_u *
                     torch.norm(self.Ru_options.R() @ x[:2], p=1),
                     torch.tensor(0, dtype=self.dtype)))
            u = torch.max(
                torch.min(u_pre_sat, torch.from_numpy(self.u_upper_limit)),
                torch.from_numpy(self.u_lower_limit))
        elif len(x.shape) == 2:
            u_pre_sat = self.controller_network(x) - self.controller_network(
                torch.zeros((3, ), dtype=self.dtype)) + torch.stack(
                    (self.lambda_u *
                     torch.norm(self.Ru_options.R() @ x[:, :2].T, p=1, dim=0),
                     torch.zeros((x.shape[0], ), dtype=self.dtype))).T
            u = torch.max(
                torch.min(
                    u_pre_sat,
                    torch.from_numpy(self.u_upper_limit).reshape((1, -1))),
                torch.from_numpy(self.u_lower_limit).reshape((1, -1)))
        return u

    def _add_network_controller_mip_constraint_given_relu_bound(
            self, prog, x_var, u_var, controller_pre_relu_lo,
            controller_pre_relu_up, network_input_lo, network_input_up,
            controller_network_output_lo, controller_network_output_up,
            controller_slack_var_name, controller_binary_var_name,
            binary_var_type):
        assert (isinstance(self.controller_network, torch.nn.Sequential))
        controller_mip_cnstr = self.controller_relu_free_pattern.\
            _output_constraint_given_bounds(
                controller_pre_relu_lo, controller_pre_relu_up,
                network_input_lo, network_input_up)
        assert (controller_mip_cnstr.Aout_input is None)
        assert (controller_mip_cnstr.Aout_binary is None)
        controller_slack, controller_binary = \
            prog.add_mixed_integer_linear_constraints(
                controller_mip_cnstr, x_var, None,
                controller_slack_var_name, controller_binary_var_name,
                "controller_ineq", "controller_eq", "", binary_var_type)

        # Write the part |Rᵤ*[p_x, p_y]|₁ = sum s
        Ru = self.Ru_options.R()
        s_dim = Ru.shape[0]
        Rx_lb, Rx_ub = mip_utils.compute_range_by_IA(
            Ru, torch.zeros((s_dim, ), dtype=self.dtype),
            torch.from_numpy(self.forward_system.x_lo_all[:2]),
            torch.from_numpy(self.forward_system.x_up_all[:2]))
        s_lb = torch.zeros((s_dim, ), dtype=self.dtype)
        s_ub = torch.zeros((s_dim, ), dtype=self.dtype)
        s = [None] * s_dim
        alpha = [None] * s_dim
        for i in range(s_dim):
            mip_cnstr_ret = utils.absolute_value_as_mixed_integer_constraint(
                Rx_lb[i], Rx_ub[i], False)
            mip_cnstr_ret.transform_input(Ru[i, :].reshape((1, -1)),
                                          torch.tensor([0], dtype=self.dtype))
            s_i, alpha_i = prog.add_mixed_integer_linear_constraints(
                mip_cnstr_ret, x_var[:2], None, "unicycle_controller_s",
                "unicycle_controller_binary", "", "", "", gurobipy.GRB.BINARY)
            assert (len(s_i) == 1)
            s[i] = s_i[0]
            assert (len(alpha_i) == 1)
            alpha[i] = alpha_i[0]

            if Rx_lb[i] < 0 and Rx_ub[i] > 0:
                s_ub[i] = torch.max(-Rx_lb[i], Rx_ub[i])
            elif Rx_lb[i] >= 0:
                s_lb[i] = Rx_lb[i]
                s_ub[i] = Rx_ub[i]
            elif Rx_ub[i] <= 0:
                s_lb[i] = -Rx_ub[i]
                s_ub[i] = -Rx_lb[i]

        u_pre_sat = prog.addVars(self.forward_system.u_dim,
                                 lb=-gurobipy.GRB.INFINITY,
                                 vtype=gurobipy.GRB.CONTINUOUS,
                                 name="u_pre_sat")
        relu_at_zero = self.controller_network(
            torch.zeros((3, ), dtype=self.dtype))
        # Add the constraint u_pre_sat = ϕ(x) − ϕ(0) + [λᵤ |Rᵤ*[p_x, p_y]|₁, 0]
        # = relu_Aout_slack * relu_slack + relu_Cout - ϕ(0) + [λᵤ * sum(s), 0]
        prog.addMConstr([
            torch.eye(2, dtype=self.dtype), -controller_mip_cnstr.Aout_slack,
            -self.lambda_u * torch.cat((torch.ones(
                (1, s_dim),
                dtype=self.dtype), torch.zeros((1, s_dim), dtype=self.dtype)),
                                       dim=0)
        ], [u_pre_sat, controller_slack, s],
                         sense=gurobipy.GRB.EQUAL,
                         b=controller_mip_cnstr.Cout - relu_at_zero)

        u_pre_sat_lo = controller_network_output_lo - relu_at_zero +\
            torch.stack((self.lambda_u * torch.sum(s_lb),
                         torch.tensor(0, dtype=self.dtype)))
        u_pre_sat_up = controller_network_output_up - relu_at_zero +\
            torch.stack((self.lambda_u * torch.sum(s_ub),
                         torch.tensor(0, dtype=self.dtype)))
        u_lower_bound, u_upper_bound = \
            feedback_system._add_input_saturation_constraint(
                prog, u_var, u_pre_sat, self.u_lower_limit, self.u_upper_limit,
                u_pre_sat_lo, u_pre_sat_up, self.dtype, binary_var_type)
        return controller_slack, controller_binary, u_lower_bound,\
            u_upper_bound, controller_pre_relu_lo, controller_pre_relu_up

    def controller_variables(self):
        return list(self.controller_network.parameters()
                    ) + self.Ru_options.variables()
