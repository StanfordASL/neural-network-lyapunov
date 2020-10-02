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

import gurobipy

import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


class FeedbackSystem:
    """
    The forward model x[n+1] = f(x[n], u[n]) can be either a
    HybridLinearSystem class, or a ReLU system.
    The feedback controller is u[n] = ϕᵤ(x[n]) - ϕᵤ(x*), where ϕᵤ is another
    neural network with (leaky) ReLU activations.
    """
    def __init__(
        self, forward_system, controller_network, x_equilibrium: torch.Tensor,
            u_equilibrium: torch.Tensor):
        """
        @param forward_system. The forward dynamical system representing
        x[n+1] = f(x[n], u[n])
        @param controller_network The network ϕᵤ, where the control law is
        u[n] = ϕᵤ(x[n]) - ϕᵤ(x*) + u*
        @param x_equilibrium The equilibrium state.
        @param u_equilibrium The control action at equilibrium.
        """
        assert(isinstance(
            forward_system, hybrid_linear_system.HybridLinearSystem) or
            isinstance(forward_system, relu_system.ReLUSystem))
        self.forward_system = forward_system
        self.x_dim = self.forward_system.x_dim
        self.x_lo_all = self.forward_system.x_lo_all
        self.x_up_all = self.forward_system.x_up_all
        self.dtype = self.forward_system.dtype
        assert(controller_network[0].in_features == self.forward_system.x_dim)
        assert(controller_network[-1].out_features ==
               self.forward_system.u_dim)
        self.controller_network = controller_network
        assert(x_equilibrium.shape == (self.forward_system.x_dim,))
        assert(x_equilibrium.dtype == self.dtype)
        self.x_equilibrium = x_equilibrium
        assert(u_equilibrium.shape == (self.forward_system.u_dim,))
        assert(u_equilibrium.dtype == self.dtype)
        self.u_equilibrium = u_equilibrium
        self.controller_relu_free_pattern = \
            relu_to_optimization.ReLUFreePattern(
                self.controller_network, self.dtype)

    def add_dynamics_mip_constraint(
        self, mip, x_var, x_next_var, u_var_name, forward_slack_var_name,
        forward_binary_var_name, controller_slack_var_name,
            controller_binary_var_name):
        assert(isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
        u = mip.addVars(
            self.forward_system.u_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name=u_var_name)
        # Now add the forward dynamics constraint
        forward_mip_cnstr = self.forward_system.mixed_integer_constraints()
        forward_slack, forward_binary = \
            mip.add_mixed_integer_linear_constraints(
                forward_mip_cnstr, x_var + u, x_next_var,
                forward_slack_var_name, forward_binary_var_name,
                "forward_dynamics_ineq", "forward_dynamics_eq",
                "forward_dynamics_output")

        # Now add the controller mip constraint.
        controller_mip_cnstr, _, _, _, _ = \
            self.controller_relu_free_pattern.output_constraint(
                torch.from_numpy(self.forward_system.x_lo_all),
                torch.from_numpy(self.forward_system.x_up_all))
        assert(controller_mip_cnstr.Aout_input is None)
        assert(controller_mip_cnstr.Aout_binary is None)
        controller_slack, controller_binary = \
            mip.add_mixed_integer_linear_constraints(
                controller_mip_cnstr, x_var, None, controller_slack_var_name,
                controller_binary_var_name, "controller_ineq", "controller_eq",
                "")
        # Add the constraint
        # u[n] = ϕᵤ(x[n]) - ϕᵤ(x*) + u*
        # Namely Aout_slack * controller_slack -u[n] = ϕᵤ(x*) - u* -Cout
        mip.addMConstrs([
            controller_mip_cnstr.Aout_slack, -torch.eye(
                self.forward_system.u_dim, dtype=self.forward_system.dtype)],
            [controller_slack, u], sense=gurobipy.GRB.EQUAL,
            b=self.controller_network(self.x_equilibrium) - self.u_equilibrium
            - controller_mip_cnstr.Cout, name="controller_output")
        return u, forward_slack, controller_slack, forward_binary,\
            controller_binary

    def compute_u(self, x):
        """
        The controller is defined as
        u[n] = ϕᵤ(x[n]) - ϕᵤ(x*) + u*
        """
        return self.controller_network(x) - \
            self.controller_network(self.x_equilibrium) + self.u_equilibrium

    def possible_dx(self, x):
        u = self.compute_u(x)
        return self.forward_system.possible_dx(x, u)

    def step_forward(self, x):
        u = self.compute_u(x)
        return self.forward_system.step_forward(x, u)[0]
