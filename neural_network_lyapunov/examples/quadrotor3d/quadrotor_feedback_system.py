import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.examples.quadrotor3d.quadrotor as quadrotor
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import gurobipy
import torch


class QuadrotorFeedbackSystem(feedback_system.FeedbackSystem):
    def __init__(self, forward_system, controller_network, u_lower_limit,
                 u_upper_limit):
        assert (isinstance(forward_system, quadrotor.QuadrotorReLUSystem))
        super(QuadrotorFeedbackSystem,
              self).__init__(forward_system,
                             controller_network,
                             forward_system.x_equilibrium,
                             forward_system.u_equilibrium,
                             u_lower_limit,
                             u_upper_limit,
                             xhat_indices=None)

    def add_dynamics_mip_constraint(self, mip, x_var, x_next_var, u_var_name,
                                    forward_slack_var_name,
                                    forward_binary_var_name,
                                    controller_slack_var_name,
                                    controller_binary_var_name):
        """
        Overloads add_dynamics_mip_constraint in the FeedbackSystem class
        When propagating the bounds through LP, we form a big LP containing
        the linear constraints in both the controller network and the
        forward network.
        """
        if self.forward_system.network_bound_propagate_method ==\
            mip_utils.PropagateBoundsMethod.IA or\
            self.controller_network_bound_propagate_method ==\
                mip_utils.PropagateBoundsMethod.IA:
            return super(QuadrotorFeedbackSystem,
                         self).add_dynamics_mip_constraint(
                             mip, x_var, x_next_var, u_var_name,
                             forward_slack_var_name, forward_binary_var_name,
                             controller_slack_var_name,
                             controller_binary_var_name)
        else:
            assert (self.forward_system.network_bound_propagate_method
                    in (mip_utils.PropagateBoundsMethod.LP,
                        mip_utils.PropagateBoundsMethod.MIP))
            assert (self.controller_network_bound_propagate_method
                    in (mip_utils.PropagateBoundsMethod.LP,
                        mip_utils.PropagateBoundsMethod.MIP))
            assert (isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
            u_var = mip.addVars(self.forward_system.u_dim,
                                lb=-gurobipy.GRB.INFINITY,
                                vtype=gurobipy.GRB.CONTINUOUS)

            # First compute the relu input bounds for the controller
            # network using LP relaxation.
            controller_network_input_lo = torch.from_numpy(
                self.forward_system.x_lo_all)
            controller_network_input_up = torch.from_numpy(
                self.forward_system.x_up_all)
            controller_relu_input_lo, controller_relu_input_up,\
                controller_relu_output_lo, controller_relu_output_up =\
                self.controller_relu_free_pattern._compute_layer_bound(
                    controller_network_input_lo,
                    controller_network_input_up,
                    self.controller_network_bound_propagate_method)
            controller_network_output_lo, controller_network_output_up =\
                self.controller_relu_free_pattern.\
                _compute_network_output_bounds(
                    controller_relu_input_lo, controller_relu_input_up,
                    controller_network_input_lo,
                    controller_network_input_up,
                    self.controller_network_bound_propagate_method)

            controller_slack, controller_binary, u_lower_bound,\
                u_upper_bound, controller_relu_input_lo,\
                controller_relu_input_up =\
                self._add_network_controller_mip_constraint_given_relu_bound(
                    mip,
                    x_var,
                    u_var,
                    controller_relu_input_lo,
                    controller_relu_input_up,
                    controller_network_input_lo,
                    controller_network_input_up,
                    controller_network_output_lo,
                    controller_network_output_up,
                    controller_slack_var_name,
                    controller_binary_var_name,
                    binary_var_type=gurobipy.GRB.BINARY)
            controller_mip_cnstr_return = \
                feedback_system.ControllerMipConstraintReturn(
                    nn_input=x_var,
                    slack=controller_slack,
                    binary=controller_binary,
                    u_lower_bound=u_lower_bound,
                    u_upper_bound=u_upper_bound,
                    relu_input_lo=controller_relu_input_lo,
                    relu_input_up=controller_relu_input_up,
                    relu_output_lo=controller_relu_output_lo,
                    relu_output_up=controller_relu_output_up)

            # Now compute the bounds on the ReLU units of the forward
            # dynamic network through LP.
            forward_network_u_lo = torch.max(self.forward_system.u_lo,
                                             u_lower_bound)
            forward_network_u_up = torch.min(self.forward_system.u_up,
                                             u_upper_bound)

            def create_prog_callback():
                prog = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
                x_dim = 12
                u_dim = 4
                x_var_lp = prog.addVars(x_dim,
                                        lb=self.forward_system.x_lo,
                                        ub=self.forward_system.x_up,
                                        vtype=gurobipy.GRB.CONTINUOUS)
                u_var_lp = prog.addVars(u_dim,
                                        lb=forward_network_u_lo,
                                        ub=forward_network_u_up,
                                        vtype=gurobipy.GRB.CONTINUOUS)
                if self.controller_network_bound_propagate_method ==\
                        mip_utils.PropagateBoundsMethod.LP:
                    binary_var_type1 = gurobipy.GRB.CONTINUOUS
                elif self.controller_network_bound_propagate_method ==\
                        mip_utils.PropagateBoundsMethod.MIP:
                    binary_var_type1 = gurobipy.GRB.BINARY
                self._add_network_controller_mip_constraint_given_relu_bound(
                    prog, x_var_lp, u_var_lp, controller_relu_input_lo,
                    controller_relu_input_up, controller_network_input_lo,
                    controller_network_input_up, controller_network_output_lo,
                    controller_network_output_up, "controller_slack",
                    "controller_binary_relax", binary_var_type1)
                forward_network_input_var = x_var_lp[3:6] + x_var_lp[
                    9:12] + u_var_lp
                return prog, forward_network_input_var

            forward_dynamics_return = \
                self.forward_system.add_dynamics_constraint(
                    mip, x_var, x_next_var, u_var, forward_slack_var_name,
                    forward_binary_var_name, additional_u_lo=u_lower_bound,
                    additional_u_up=u_upper_bound,
                    create_lp_prog_callback=create_prog_callback)
            return u_var, forward_dynamics_return, controller_mip_cnstr_return
