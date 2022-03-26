import neural_network_lyapunov.feedback_system as feedback_system
import gurobipy
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


def _add_system_constraint(system,
                           milp,
                           x,
                           x_next,
                           *,
                           binary_var_type=gurobipy.GRB.BINARY):
    """
    This function is intended for internal usage only (but I expose it
    as a public function for unit test).
    Add the constraint and variables to write the hybrid linear system
    dynamics as mixed-integer linear constraints.
    @param binary_var_type Refer to GurobiTorchMIP.addVars for more
    details.
    """
    if isinstance(
        system, hybrid_linear_system.AutonomousHybridLinearSystem)\
            or isinstance(system, relu_system.AutonomousReLUSystem)\
            or isinstance(
                system, relu_system.AutonomousReLUSystemGivenEquilibrium)\
            or isinstance(
                system,
                relu_system.AutonomousResidualReLUSystemGivenEquilibrium):
        assert (isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        return system.add_dynamics_constraint(milp, x, x_next, "s", "gamma",
                                              binary_var_type)

    elif isinstance(system, feedback_system.FeedbackSystem):
        u, forward_dynamics_return, controller_mip_cnstr_return = \
            system.add_dynamics_mip_constraint(
                milp, x, x_next, "u", "forward_s", "forward_binary",
                "controller_s", "controller_binary", binary_var_type)
        slack = u + forward_dynamics_return.slack +\
            controller_mip_cnstr_return.slack
        binary = forward_dynamics_return.binary + \
            controller_mip_cnstr_return.binary
        ret = hybrid_linear_system.DynamicsConstraintReturn(slack, binary)
        ret.x_next_lb_IA = forward_dynamics_return.x_next_lb_IA
        ret.x_next_ub_IA = forward_dynamics_return.x_next_ub_IA
        ret.x_next_bound_prog = forward_dynamics_return.x_next_bound_prog
        ret.x_next_bound_var = forward_dynamics_return.x_next_bound_var
        ret.x_var = forward_dynamics_return.x_var
        ret.forward_dynamics_return = forward_dynamics_return
        ret.controller_mip_cnstr_return = controller_mip_cnstr_return
        return ret
    else:
        raise (NotImplementedError)
