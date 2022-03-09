"""
For a given controller piecewise affine dynamical system x[n+1] = f(x[n],u[n]),
(represented either by a neural network with ReLU activation functions, or a
HybridLinearSystem), suppose the equilibrium of this system is x*, namely
x* = f(x*, u*)
we construct a piecewise affine controller
u[n] = ϕᵤ(x[n]) - ϕᵤ(x*) + u*
where ϕᵤ(x) is a neural network with ReLU activation functions. This feedback
system is still a piecewise affine system. We can represent the relationship
between x[n], x[n+1] and u[n] with mixed-integer linear constraints.
"""

import torch
import numpy as np

import gurobipy

import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils


class ControllerMipConstraintReturn:
    def __init__(self, nn_input, slack, binary, u_lower_bound, u_upper_bound,
                 relu_input_lo, relu_input_up, relu_output_lo, relu_output_up,
                 control_bound_prog: relu_system.ControlBoundProg):
        self.nn_input = nn_input
        self.slack = slack
        self.binary = binary
        self.u_lower_bound = u_lower_bound
        self.u_upper_bound = u_upper_bound
        self.relu_input_lo = relu_input_lo
        self.relu_input_up = relu_input_up
        self.relu_output_lo = relu_output_lo
        self.relu_output_up = relu_output_up
        self.control_bound_prog = control_bound_prog


class FeedbackSystem:
    """
    The forward model x[n+1] = f(x[n], u[n]) can be either a
    HybridLinearSystem class, or a ReLU system.
    The feedback controller is u[n] = ϕᵤ(x[n]) - ϕᵤ(x*), where ϕᵤ is another
    neural network with (leaky) ReLU activations.
    """
    def __init__(self, forward_system, controller_network,
                 x_equilibrium: torch.Tensor, u_equilibrium: torch.Tensor,
                 u_lower_limit: np.ndarray, u_upper_limit: np.ndarray):
        """
        @param forward_system. The forward dynamical system representing
        x[n+1] = f(x[n], u[n]). This system must implements functions like
        add_dynamics_constraint(). Check ReLUSystemGivenEquilibrium in
        relu_system.py as a reference.
        @param controller_network The network ϕᵤ, where the control law is
        u[n] = ϕᵤ(x[n]) - ϕᵤ(x*) + u*
        @param x_equilibrium The equilibrium state.
        @param u_equilibrium The control action at equilibrium.
        @param u_lower_limit The lower limit for the control u[n]. We will
        saturate the control if it is below u_lower_limit. Set to
        u_lower_limit[i] to -infinity if you don't want saturation for the i'th
        control.
        @param u_upper_limit The upper limit for the control u[n]. We will
        saturate the control if it is above u_upper_limit. Set to
        u_upper_limit[i] to infinity if you don't want saturation for the i'th
        control.
        @note If a control has a lower limit, it has to also have an upper
        limit, and vice versa.
        """
        self.forward_system = forward_system
        self.x_dim = self.forward_system.x_dim
        self.x_lo_all = self.forward_system.x_lo_all
        self.x_up_all = self.forward_system.x_up_all
        self.dtype = self.forward_system.dtype
        if isinstance(controller_network, torch.nn.Sequential):
            assert (
                controller_network[0].in_features == self.forward_system.x_dim)
            assert (controller_network[-1].out_features ==
                    self.forward_system.u_dim)
        elif isinstance(controller_network, torch.nn.Linear):
            assert (
                controller_network.in_features == self.forward_system.x_dim)
            assert (
                controller_network.out_features == self.forward_system.u_dim)
        else:
            raise Exception("Unknown controller type.")
        self.controller_network = controller_network
        assert (x_equilibrium.shape == (self.forward_system.x_dim, ))
        assert (x_equilibrium.dtype == self.dtype)
        self.x_equilibrium = x_equilibrium
        assert (u_equilibrium.shape == (self.forward_system.u_dim, ))
        assert (u_equilibrium.dtype == self.dtype)
        self.u_equilibrium = u_equilibrium
        if isinstance(self.controller_network, torch.nn.Sequential):
            self.controller_relu_free_pattern = \
                relu_to_optimization.ReLUFreePattern(
                    self.controller_network, self.dtype)
        assert (isinstance(u_lower_limit, np.ndarray))
        assert (u_lower_limit.shape == (self.forward_system.u_dim, ))
        assert (isinstance(u_upper_limit, np.ndarray))
        assert (u_upper_limit.shape == (self.forward_system.u_dim, ))
        self.u_lower_limit = u_lower_limit
        self.u_upper_limit = u_upper_limit
        self.controller_network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.IA

    def _add_network_controller_mip_constraint_given_relu_bound(
            self, prog, x_var, u_var, controller_pre_relu_lo,
            controller_pre_relu_up, network_input_lo, network_input_up,
            controller_network_output_lo, controller_network_output_up,
            controller_slack_var_name, controller_binary_var_name,
            binary_var_type):
        assert (isinstance(self.controller_network, torch.nn.Sequential))
        controller_mip_cnstr =\
            self.controller_relu_free_pattern._output_constraint_given_bounds(
                controller_pre_relu_lo, controller_pre_relu_up,
                network_input_lo, network_input_up)
        assert (controller_mip_cnstr.Aout_input is None)
        assert (controller_mip_cnstr.Aout_binary is None)
        controller_slack, controller_binary = \
            prog.add_mixed_integer_linear_constraints(
                controller_mip_cnstr, x_var, None,
                controller_slack_var_name, controller_binary_var_name,
                "controller_ineq", "controller_eq", "", binary_var_type)
        u_pre_sat = prog.addVars(self.forward_system.u_dim,
                                 lb=-gurobipy.GRB.INFINITY,
                                 vtype=gurobipy.GRB.CONTINUOUS,
                                 name="u_pre_sat")
        # compute ϕᵤ(x*)
        relu_x_equilibrium = self.controller_network(self.x_equilibrium)

        # Add the input saturation constraint
        # u_pre_sat = ϕᵤ(x[n]) - ϕᵤ(x*) + u*
        # and u[n] = saturation(u_pre_sat)
        # Then the constraint is
        # Aout_slack * controller_slack -u_pre_sat
        # = relu_x_equilibrium- u* -Cout
        if isinstance(self.controller_network, torch.nn.Sequential):
            prog.addMConstrs([
                controller_mip_cnstr.Aout_slack.reshape(
                    (self.forward_system.u_dim, len(controller_slack))),
                -torch.eye(self.forward_system.u_dim,
                           dtype=self.forward_system.dtype)
            ], [controller_slack, u_pre_sat],
                             sense=gurobipy.GRB.EQUAL,
                             b=relu_x_equilibrium - self.u_equilibrium -
                             controller_mip_cnstr.Cout,
                             name="controller_output")

        u_pre_sat_lo = controller_network_output_lo -\
            relu_x_equilibrium + self.u_equilibrium
        u_pre_sat_up = controller_network_output_up -\
            relu_x_equilibrium + self.u_equilibrium
        u_lower_bound, u_upper_bound = _add_input_saturation_constraint(
            prog, u_var, u_pre_sat, self.u_lower_limit, self.u_upper_limit,
            u_pre_sat_lo, u_pre_sat_up, self.dtype, binary_var_type)
        # Note that controller_binary only contains the binary variables in
        # the relu network. It doesn't contain the binary variables in the
        # saturation block.
        return controller_slack, controller_binary, u_lower_bound,\
            u_upper_bound, controller_pre_relu_lo, controller_pre_relu_up

    def _add_network_controller_mip_constraint(self, mip, x_var, u_var,
                                               controller_slack_var_name,
                                               controller_binary_var_name,
                                               binary_var_type):
        network_input_lo = torch.from_numpy(self.forward_system.x_lo_all)
        network_input_up = torch.from_numpy(self.forward_system.x_up_all)
        controller_pre_relu_lo, controller_pre_relu_up,\
            controller_post_relu_lo, controller_post_relu_up, =\
            self.controller_relu_free_pattern._compute_layer_bound(
                network_input_lo, network_input_up,
                self.controller_network_bound_propagate_method)
        network_output_lo, network_output_up =\
            self.controller_relu_free_pattern._compute_network_output_bounds(
                controller_pre_relu_lo, controller_pre_relu_up,
                network_input_lo, network_input_up,
                self.controller_network_bound_propagate_method)

        controller_slack, controller_binary, u_lower_bound, u_upper_bound,\
            controller_pre_relu_lo, controller_pre_relu_up =\
            self._add_network_controller_mip_constraint_given_relu_bound(
                mip, x_var, u_var, controller_pre_relu_lo,
                controller_pre_relu_up, network_input_lo, network_input_up,
                network_output_lo, network_output_up,
                controller_slack_var_name, controller_binary_var_name,
                binary_var_type)

        control_bound_prog = relu_system.ControlBoundProg(None, None, None)
        control_bound_prog.prog = gurobi_torch_mip.GurobiTorchMILP(self.dtype)
        control_bound_prog.x_var = control_bound_prog.prog.addVars(
            self.x_dim, lb=-gurobipy.GRB.INFINITY)
        control_bound_prog.u_var = control_bound_prog.prog.addVars(
            self.forward_system.u_dim, lb=-gurobipy.GRB.INFINITY)

        if self.controller_network_bound_propagate_method != \
                mip_utils.PropagateBoundsMethod.IA:
            self._add_network_controller_mip_constraint_given_relu_bound(
                control_bound_prog.prog, control_bound_prog.x_var,
                control_bound_prog.u_var, controller_pre_relu_lo,
                controller_pre_relu_up, network_input_lo, network_input_up,
                network_output_lo, network_output_up, "", "",
                mip_utils.binary_var_type_per_method(
                    self.controller_network_bound_propagate_method))
        return ControllerMipConstraintReturn(
            nn_input=x_var,
            slack=controller_slack,
            binary=controller_binary,
            u_lower_bound=u_lower_bound,
            u_upper_bound=u_upper_bound,
            relu_input_lo=controller_pre_relu_lo,
            relu_input_up=controller_pre_relu_up,
            relu_output_lo=controller_post_relu_lo,
            relu_output_up=controller_post_relu_up,
            control_bound_prog=control_bound_prog)

    def _add_linear_controller_mip_constraint(self, mip, x_var, u_var,
                                              binary_var_type):
        assert (isinstance(self.controller_network, torch.nn.Linear))
        controller_slack = []
        controller_binary = []
        u_pre_sat = mip.addVars(self.forward_system.u_dim,
                                lb=-gurobipy.GRB.INFINITY,
                                vtype=gurobipy.GRB.CONTINUOUS,
                                name="u_pre_sat")

        network_at_x_equilibrium = self.controller_network(self.x_equilibrium)
        # Add the input saturation constraint
        # u_pre_sat = ϕᵤ(x[n]) - ϕᵤ(x*) + u*
        # and u[n] = saturation(u_pre_sat)
        # The constraint is
        # W * x[n] -u_pre_sat =ϕᵤ(x*) - u* - bias
        if self.controller_network.bias is not None:
            bias = self.controller_network.bias
        else:
            bias = torch.zeros((self.forward_system.u_dim, ), dtype=self.dtype)
        mip.addMConstrs([
            -torch.eye(self.forward_system.u_dim, dtype=self.dtype),
            self.controller_network.weight
        ], [u_pre_sat, x_var],
                        b=-self.u_equilibrium + network_at_x_equilibrium -
                        bias,
                        sense=gurobipy.GRB.EQUAL,
                        name="controller")
        controller_network_output_lo, controller_network_output_up =\
            mip_utils.propagate_bounds(
                self.controller_network,
                torch.from_numpy(self.forward_system.x_lo_all),
                torch.from_numpy(self.forward_system.x_up_all))
        u_pre_sat_lo = controller_network_output_lo -\
            network_at_x_equilibrium + self.u_equilibrium
        u_pre_sat_up = controller_network_output_up -\
            network_at_x_equilibrium + self.u_equilibrium
        u_lower_bound, u_upper_bound = _add_input_saturation_constraint(
            mip, u_var, u_pre_sat, self.u_lower_limit, self.u_upper_limit,
            u_pre_sat_lo, u_pre_sat_up, self.dtype, binary_var_type)
        return controller_slack, controller_binary, u_lower_bound,\
            u_upper_bound

    def _add_controller_mip_constraint(self, mip, x_var, u_var,
                                       controller_slack_var_name,
                                       controller_binary_var_name,
                                       binary_var_type):
        if isinstance(self.controller_network, torch.nn.Sequential):

            return self._add_network_controller_mip_constraint(
                mip, x_var, u_var, controller_slack_var_name,
                controller_binary_var_name, binary_var_type)
        elif isinstance(self.controller_network, torch.nn.Linear):
            controller_slack, controller_binary, u_lower_bound, u_upper_bound\
                = self._add_linear_controller_mip_constraint(
                    mip, x_var, u_var, binary_var_type)
            control_bound_prog = relu_system.ControlBoundProg(None, None, None)
            control_bound_prog.prog = gurobi_torch_mip.GurobiTorchMILP(
                self.dtype)
            control_bound_prog.x_var = control_bound_prog.prog.addVars(
                self.x_dim, lb=-gurobipy.GRB.INFINITY)
            control_bound_prog.u_var = control_bound_prog.prog.addVars(
                self.forward_system.u_dim, lb=-gurobipy.GRB.INFINITY)
            if self.controller_network_bound_propagate_method != \
                    mip_utils.PropagateBoundsMethod.IA:
                self._add_linear_controller_mip_constraint(
                    control_bound_prog.prog, control_bound_prog.x_var,
                    control_bound_prog.u_var,
                    mip_utils.binary_var_type_per_method(
                        self.controller_network_bound_propagate_method))
            return ControllerMipConstraintReturn(
                nn_input=None,
                slack=controller_slack,
                binary=controller_binary,
                u_lower_bound=u_lower_bound,
                u_upper_bound=u_upper_bound,
                relu_input_lo=None,
                relu_input_up=None,
                relu_output_lo=None,
                relu_output_up=None,
                control_bound_prog=control_bound_prog)

    def strengthen_controller_mip_constraint(
            self, mip: gurobi_torch_mip.GurobiTorchMIP, x_var: list,
            controller_slack: list, controller_binary: list,
            controller_post_relu_lo, controller_post_relu_up):
        """
        Given the variables x_var, controller_slack, controller_binary (and the
        value of these variables stored inside x_var, controller_slack,
        controller_binary), we strengthen the controller mip constraint by
        adding the most violated ideal constraint at the value stored inside
        x_var, controller_slack and controller_binary.
        Note that after calling this function, @param mip might get changed, it
        will contain the strengthened constraint if that constraint is violated
        at the values in x_var, controller_slack, controller_binary.
        """
        assert (isinstance(self.controller_network, torch.nn.Sequential))
        assert (isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
        assert (isinstance(x_var, list))
        assert (isinstance(controller_slack, list))
        assert (isinstance(controller_binary, list))
        assert (isinstance(controller_post_relu_lo, torch.Tensor))
        assert (isinstance(controller_post_relu_up, torch.Tensor))
        linear_inputs = torch.tensor([v.x for v in x_var] +
                                     [v.x for v in controller_slack],
                                     dtype=mip.dtype)
        relu_activations = torch.tensor([v.x for v in controller_binary],
                                        dtype=mip.dtype)
        linear_inputs_lo = torch.cat(
            (torch.from_numpy(self.forward_system.x_lo_all),
             controller_post_relu_lo))
        linear_inputs_up = torch.cat(
            (torch.from_numpy(self.forward_system.x_up_all),
             controller_post_relu_up))
        Ain_x, Ain_slack, Ain_binary, rhs_in = \
            self.controller_relu_free_pattern.strengthen_mip_at_point(
                (linear_inputs, relu_activations), linear_inputs_lo,
                linear_inputs_up)
        if Ain_x is not None:
            mip.addMConstrs([Ain_x, Ain_slack, Ain_binary],
                            [x_var, controller_slack, controller_binary],
                            b=rhs_in,
                            sense=gurobipy.GRB.LESS_EQUAL,
                            name="controller relu")

    def add_dynamics_mip_constraint(self,
                                    mip,
                                    x_var,
                                    x_next_var,
                                    u_var_name,
                                    forward_slack_var_name,
                                    forward_binary_var_name,
                                    controller_slack_var_name,
                                    controller_binary_var_name,
                                    binary_var_type=gurobipy.GRB.BINARY):
        assert (isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
        u = mip.addVars(self.forward_system.u_dim,
                        lb=-gurobipy.GRB.INFINITY,
                        vtype=gurobipy.GRB.CONTINUOUS,
                        name=u_var_name)
        # Add the controller constraint.
        controller_mip_cnstr_return = \
            self._add_controller_mip_constraint(
                mip, x_var, u, controller_slack_var_name,
                controller_binary_var_name, binary_var_type)

        # Now add the forward dynamics constraint
        forward_dynamics_return = \
            self.forward_system.add_dynamics_constraint(
                mip, x_var, x_next_var, u, forward_slack_var_name,
                forward_binary_var_name,
                additional_u_lo=controller_mip_cnstr_return.u_lower_bound,
                additional_u_up=controller_mip_cnstr_return.u_upper_bound,
                binary_var_type=binary_var_type,
                u_input_prog=controller_mip_cnstr_return.control_bound_prog)

        return u, forward_dynamics_return, controller_mip_cnstr_return

    def strengthen_dynamics_constraint(
            self,
            mip: gurobi_torch_mip.GurobiTorchMIP,
            forward_dynamics_return: hybrid_linear_system.
        DynamicsConstraintReturn,  # noqa
            controller_mip_cnstr_return: ControllerMipConstraintReturn):
        """
        Strengthen the MIP constraint on system dynamics.
        For ReLU network, we can strengthen its MIP constraint (derived from
        big-M technique), by adding the mostly violated ideal constraint
        evaluated at a point.
        @param forward_dynamics_return Returned from
        add_dynamics_mip_constraint()
        @param controller_mip_cnstr_return: Returned from
        add_dynamics_mip_constraint()
        """
        assert (isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
        assert (isinstance(forward_dynamics_return,
                           hybrid_linear_system.DynamicsConstraintReturn))
        assert (isinstance(controller_mip_cnstr_return,
                           ControllerMipConstraintReturn))
        if (isinstance(forward_dynamics_return,
                       relu_system.ReLUDynamicsConstraintReturn)):
            # Value of the inputs to each linear layer in the forward dynamics
            # network.
            forward_nn_linear_inputs = torch.tensor(
                [v.x for v in forward_dynamics_return.nn_input] +
                [v.x for v in forward_dynamics_return.slack],
                dtype=mip.dtype)
            forward_relu_activations = torch.tensor(
                [v.x for v in forward_dynamics_return.binary], dtype=mip.dtype)
            forward_nn_linear_inputs_lo = torch.cat(
                (forward_dynamics_return.nn_input_lo,
                 forward_dynamics_return.relu_output_lo))
            forward_nn_linear_inputs_up = torch.cat(
                (forward_dynamics_return.nn_input_up,
                 forward_dynamics_return.relu_output_up))
            Ain_forward_nn_input, Ain_forward_slack, Ain_forward_binary,\
                rhs_in_forward = self.forward_system.\
                dynamics_relu_free_pattern.strengthen_mip_at_point(
                    (forward_nn_linear_inputs, forward_relu_activations),
                    forward_nn_linear_inputs_lo, forward_nn_linear_inputs_up)
            if Ain_forward_nn_input is not None:
                mip.addMConstrs([
                    Ain_forward_nn_input, Ain_forward_slack, Ain_forward_binary
                ], [
                    forward_dynamics_return.nn_input,
                    forward_dynamics_return.slack,
                    forward_dynamics_return.binary
                ],
                                b=rhs_in_forward,
                                sense=gurobipy.GRB.LESS_EQUAL,
                                name="forward relu strengthened")

        if (controller_mip_cnstr_return.nn_input is not None):
            self.strengthen_controller_mip_constraint(
                mip, controller_mip_cnstr_return.nn_input,
                controller_mip_cnstr_return.slack,
                controller_mip_cnstr_return.binary,
                controller_mip_cnstr_return.relu_output_lo,
                controller_mip_cnstr_return.relu_output_up)

    def compute_u(self, x):
        """
        The controller is defined as
        u[n] = ϕᵤ(x[n]) - ϕᵤ(x*) + u*
        """
        u_pre_sat = self.controller_network(x) - \
            self.controller_network(self.x_equilibrium) + self.u_equilibrium
        if len(x.shape) == 1:
            u = torch.max(
                torch.min(u_pre_sat, torch.from_numpy(self.u_upper_limit)),
                torch.from_numpy(self.u_lower_limit))
        else:
            # batch of x
            u = torch.max(
                torch.min(
                    u_pre_sat,
                    torch.from_numpy(self.u_upper_limit).reshape((1, -1))),
                torch.from_numpy(self.u_lower_limit).reshape((1, -1)))
        return u

    def possible_dx(self, x):
        u = self.compute_u(x)
        return self.forward_system.possible_dx(x, u)

    def step_forward(self, x):
        u = self.compute_u(x)
        if isinstance(self.forward_system,
                      hybrid_linear_system.HybridLinearSystem):
            return self.forward_system.step_forward(x, u)[0]
        else:
            return self.forward_system.step_forward(x, u)

    def controller_variables(self):
        return list(self.controller_network.parameters())


def _add_input_saturation_constraint(mip, u_var, u_pre_sat,
                                     u_lower_limit: np.ndarray,
                                     u_upper_limit: np.ndarray,
                                     u_pre_sat_lo: torch.Tensor,
                                     u_pre_sat_up: torch.Tensor, dtype,
                                     binary_var_type):
    """
    Add the MIP constraints of the saturation block. Also output the bounds of
    u after the saturation.
    """
    assert (np.all(u_lower_limit <= u_upper_limit))
    assert (torch.all(u_pre_sat_lo <= u_pre_sat_up))
    u_lower_bound = torch.min(
        torch.max(torch.from_numpy(u_lower_limit), u_pre_sat_lo),
        torch.from_numpy(u_upper_limit))
    u_upper_bound = torch.max(
        torch.min(torch.from_numpy(u_upper_limit), u_pre_sat_up),
        torch.from_numpy(u_lower_limit))
    for i in range(len(u_var)):
        if np.isinf(u_lower_limit[i]) and\
                np.isinf(u_upper_limit[i]):
            mip.addLConstr([torch.tensor([1, -1], dtype=dtype)],
                           [[u_var[i], u_pre_sat[i]]],
                           rhs=0.,
                           sense=gurobipy.GRB.EQUAL)
        else:
            assert (not np.isinf(u_lower_limit[i])
                    and not np.isinf(u_upper_limit[i]))
            utils.add_saturation_as_mixed_integer_constraint(
                mip, u_pre_sat[i], u_var[i], u_lower_limit[i],
                u_upper_limit[i], u_pre_sat_lo[i], u_pre_sat_up[i],
                binary_var_type)
    assert (torch.all(u_lower_bound <= u_upper_bound))
    return u_lower_bound, u_upper_bound
