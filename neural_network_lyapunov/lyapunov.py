# -*- coding: utf-8 -*-
import gurobipy
import torch
import numpy as np

import queue
from enum import Enum

import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils


class ConvergenceEps(Enum):
    """
    The epsilon constant in the Lyapunov derivative has different
    interpretations, it could represent the exponential convergence rate,
    where ε_min ≤ −V̇/V ≤ ε_max, or it can represent the asymptotic
    convergence, namely V̇ ≤ −ε |x−x*|₁.
    """
    # Exponential convergence rate upper bound.
    ExpUpper = 1
    # Exponential convergence rate lower bound.
    ExpLower = 2
    # Asymptotic convergence.
    Asymp = 3


class LyapunovHybridLinearSystem:
    """
    This is the super class of LyapunovDiscreteTimeHybridSystem and
    LyapunovContinuousTimeHybridSystem. It implements the common part of these
    two subclasses.
    """

    def __init__(self, system, lyapunov_relu):
        """
        @param system A AutonomousHybridLinearSystem or AutonomousReLUSystem
        instance.
        @param lyapunov_relu A ReLU network used to represent the Lyapunov
        function. The Lyapunov function is ReLU(x) - ReLU(x*) + λ|x-x*|₁
        """
        assert(
            isinstance(
                system, hybrid_linear_system.AutonomousHybridLinearSystem)
            or
            isinstance(
                system, relu_system.AutonomousReLUSystem)
            or
            isinstance(
                system, relu_system.AutonomousReLUSystemGivenEquilibrium)
            or
            isinstance(
                system,
                relu_system.AutonomousResidualReLUSystemGivenEquilibrium)
            or isinstance(system, feedback_system.FeedbackSystem)
        )
        self.system = system
        self.lyapunov_relu = lyapunov_relu
        self.lyapunov_relu_free_pattern = \
            relu_to_optimization.ReLUFreePattern(
                lyapunov_relu, self.system.dtype)

    def add_system_constraint(self, milp, x, x_next):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add the constraint and variables to write the hybrid linear system
        dynamics as mixed-integer linear constraints.
        """
        if isinstance(
            self.system, hybrid_linear_system.AutonomousHybridLinearSystem)\
                or isinstance(self.system, relu_system.AutonomousReLUSystem)\
                or isinstance(
                    self.system,
                    relu_system.AutonomousReLUSystemGivenEquilibrium)\
                or isinstance(
                    self.system,
                    relu_system.AutonomousResidualReLUSystemGivenEquilibrium):
            assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
            mip_cnstr_return = self.system.mixed_integer_constraints()
            s, gamma = milp.add_mixed_integer_linear_constraints(
                mip_cnstr_return, x, x_next, "s", "gamma",
                "hybrid_ineq_dynamics", "hybrid_eq_dynamics",
                "hybrid_output_dynamics")
            return s, gamma

        elif isinstance(self.system, feedback_system.FeedbackSystem):
            u, forward_slack, controller_slack, forward_binary,\
                controller_binary = self.system.add_dynamics_mip_constraint(
                    milp, x, x_next, "u", "forward_s", "forward_binary",
                    "controller_s", "controller_binary")
            slack = u + forward_slack + controller_slack
            binary = forward_binary + controller_binary
            return slack, binary
        else:
            raise(NotImplementedError)

    def add_relu_output_constraint(
            self, milp, x, slack_name="relu_z", binary_var_name="relu_beta"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add the relu output as mixed-integer linear constraint.
        @return (z, beta, a_out, b_out) z is the continuous slack variable.
        beta is the binary variable indicating whether a (leaky) ReLU unit is
        active or not. The output of the network can be written as
        a_out.dot(z) + b_out
        """
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        assert(isinstance(x, list))
        mip_constr_return, _, _, _, _ = \
            self.lyapunov_relu_free_pattern.output_constraint(
                 torch.from_numpy(self.system.x_lo_all),
                 torch.from_numpy(self.system.x_up_all))
        relu_z, relu_beta = milp.add_mixed_integer_linear_constraints(
            mip_constr_return, x, None, slack_name, binary_var_name,
            "milp_relu_ineq", "milp_relu_eq", "")
        return (relu_z, relu_beta, mip_constr_return.Aout_slack,
                mip_constr_return.Cout)

    def add_state_error_l1_constraint(
            self, milp, x_equilibrium, x, slack_name="s",
            binary_var_name="alpha"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add the L1 loss |x-x*|₁ as mixed-integer linear constraints.
        return (s, alpha) s is the continuous slack variable,
        s(i) = |x(i) - x*(i)|, alpha is the binary variable,
        alpha(i) = 1 => x(i) >= x*(i), alpha(i) = 0 => x(i)<=x*(i).
        """
        if not torch.all(torch.from_numpy(self.system.x_lo_all) <=
                         x_equilibrium) or\
                not torch.all(torch.from_numpy(self.system.x_up_all) >=
                              x_equilibrium):
            raise Exception("add_state_error_l1_constraint: we currently " +
                            "require that x_lo <= x_equilibrium <= x_up")
        s = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name=slack_name)
        alpha = milp.addVars(
            self.system.x_dim, vtype=gurobipy.GRB.BINARY, name=binary_var_name)

        for i in range(self.system.x_dim):
            if self.system.x_lo_all[i] < x_equilibrium[i] and\
                    self.system.x_up_all[i] > x_equilibrium[i]:
                Ain_x, Ain_s, Ain_alpha, rhs_in =\
                    utils.replace_absolute_value_with_mixed_integer_constraint(
                        self.system.x_lo_all[i] - x_equilibrium[i],
                        self.system.x_up_all[i] - x_equilibrium[i],
                        dtype=self.system.dtype)
                milp.addMConstrs(
                    [Ain_x.reshape((-1, 1)), Ain_s.reshape((-1, 1)),
                     Ain_alpha.reshape((-1, 1))],
                    [[x[i]], [s[i]], [alpha[i]]],
                    sense=gurobipy.GRB.LESS_EQUAL,
                    b=rhs_in + Ain_x * x_equilibrium[i])
            elif self.system.x_lo_all[i] >= x_equilibrium[i]:
                # x_lo[i] >= x*[i], so s[i] = x[i] - x*[i], alpha[i] = 1
                milp.addMConstrs(
                    [torch.tensor([[1., -1., 0], [0, 0, 1]],
                                  dtype=self.system.dtype)],
                    [[x[i], s[i], alpha[i]]], sense=gurobipy.GRB.EQUAL,
                    b=torch.stack([
                        x_equilibrium[i],
                        torch.tensor(1, dtype=self.system.dtype)]))
            else:
                # x_up[i] <= x*[i], so s[i] = x*[i] - x[i], alpha[i] = 0
                milp.addMConstrs(
                    [torch.tensor(
                        [[1, 1, 0], [0, 0, 1]], dtype=self.system.dtype)],
                    [[x[i], s[i], alpha[i]]], sense=gurobipy.GRB.EQUAL,
                    b=torch.stack([
                        x_equilibrium[i],
                        torch.tensor(0, dtype=self.system.dtype)]))

        return (s, alpha)

    def lyapunov_value(
            self, x, x_equilibrium, V_lambda,
            relu_at_equilibrium=None):
        """
        Compute the value of the Lyapunov function as
        ReLU(x) - ReLU(x*) + λ|x-x*|₁
        where |x-x*|₁ is the 1-norm of x-x*.
        @param x a torch tensor. Evaluate Lyapunov function at this point.
        @param x_equilibrium a torch tensor. The equilibrium state x*.
        @param V_lambda λ in the documentation above.
        @param relu_at_equilibrium. ReLU(x*). If set to None, then we compute
        ReLU(x*) in this function.
        """
        if relu_at_equilibrium is None:
            relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        if x.shape == (self.system.x_dim,):
            # A single state.
            return self.lyapunov_relu.forward(x) - relu_at_equilibrium +\
                V_lambda * torch.norm(x - x_equilibrium, p=1)
        else:
            # A batch of states.
            assert(x.shape[1] == self.system.x_dim)
            return self.lyapunov_relu(x).squeeze() - relu_at_equilibrium + \
                V_lambda * torch.norm(x - x_equilibrium, p=1, dim=1)

    def lyapunov_positivity_as_milp(
            self, x_equilibrium, V_lambda, V_epsilon, x_warmstart=None):
        """
        For a ReLU network, in order to determine if the function
        V(x) = ReLU(x) - ReLU(x*) + λ * |x - x*|₁
        where |x - x*|₁ is the 1-norm of the vector x - x*.
        satisfies the positivity constraint of Lyapunov condition
        V(x) > 0 ∀ x ≠ x*
        We check a strong condition
        V(x) ≥ ε |x - x*|₁ ∀ x
        where ε is a small positive number.To check if the stronger condition
        is satisfied, we can solve the following optimization problem
        max x  (ε-λ) * |x - x*|₁ - ReLU(x) + ReLU(x*)
        We can formulate this optimization problem as a mixed integer linear
        program, solve the for optimal solution of this program. If the optimal
        cost is no larger than 0, then we proved the positivity constraint
        V(x) > 0 ∀ x ≠ x*
        @param x_equilibrium The equilibrium state x*.
        @param V_lambda ρ in the documentation above.
        @param V_epsilon A scalar. ε in the documentation above.
        @return (milp, x) milp is a GurobiTorchMILP instance, x is the decision
        variable for state.
        @param x_warmstart tensor of size self.system.x_dim. If provided, will
        use x_warmstart as initial guess for the *binary* variables of the
        milp. Instead of warm start beta with the binary variable solution from
        the previous iteration, we choose to recompute beta using the previous
        adversarial state `x` in the current neural network, so as to make
        sure that this initial guess of beta is always a feasible solution.
        """
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (self.system.x_dim,))
        assert(isinstance(V_lambda, float))
        assert(isinstance(V_epsilon, float))
        if x_warmstart is not None:
            assert(isinstance(x_warmstart, torch.Tensor))
            assert(x_warmstart.shape == (self.system.x_dim,))

        dtype = self.system.dtype
        milp = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x")
        # z is the slack variable to write the output of ReLU network as mixed
        # integer constraints.
        (z, beta, a_out, b_out) = self.add_relu_output_constraint(milp, x)

        # warmstart the binary variables
        if x_warmstart is not None:
            relu_to_optimization.set_activation_warmstart(
                self.lyapunov_relu, beta, x_warmstart)

        # Now compute ReLU(x*)
        relu_x_equilibrium = self.lyapunov_relu.forward(x_equilibrium)

        # Now write the 1-norm |x - x*|₁ as mixed-integer linear constraints.
        (s, gamma) = self.add_state_error_l1_constraint(
            milp, x_equilibrium, x, slack_name="s", binary_var_name="gamma")

        milp.setObjective(
            [-a_out,
             (V_epsilon-V_lambda) *
             torch.ones((self.system.x_dim,), dtype=dtype)],
            [z, s], constant=-b_out + relu_x_equilibrium.squeeze(),
            sense=gurobipy.GRB.MAXIMIZE)
        return (milp, x)

    def lyapunov_positivity_loss_at_samples(
            self, relu_at_equilibrium, x_equilibrium,
            state_samples, V_lambda, epsilon, margin=0.):
        """
        We will sample a state xⁱ, and we would like the Lyapunov function to
        be larger than 0 at xⁱ. Hence we define the loss as
        mean(max(-V(xⁱ) + ε |xⁱ - x*|₁ + margin, 0))
        @param relu_at_equilibrium A 0-D tensor. ReLU(x*)
        @param x_equilibrium x* in the documentation above.
        @param state_samples A batch of sampled states, state_samples[i] is
        the i'th sample xⁱ.
        @param V_lambda λ in the documentation above.
        @param epsilon ε in the documentation above.
        @param margin The margin used in the hinge loss.
        """
        assert(isinstance(relu_at_equilibrium, torch.Tensor))
        assert(isinstance(state_samples, torch.Tensor))
        assert(state_samples.shape[1] == self.system.x_dim)
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (self.system.x_dim,))
        assert(isinstance(V_lambda, float))
        assert(isinstance(margin, float))
        return torch.nn.HingeEmbeddingLoss(margin=margin)(
            self.lyapunov_value(
                state_samples, x_equilibrium, V_lambda,
                relu_at_equilibrium) - epsilon * torch.norm(
                    state_samples - x_equilibrium, p=1, dim=1),
            torch.tensor(-1.).to(state_samples.device))

    def add_lyapunov_bounds_constraint(
        self, lyapunov_lower, lyapunov_upper, milp, a_relu, b_relu, V_lambda,
            relu_x_equilibrium, relu_z, state_error_s):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add constraint lower <= V(x) <= upper to @p milp, where the Lyapunov
        function V(x) = ReLU(x) - ReLU(x*) + λ|x-x*|₁.
        Also we have ReLU(x) = a_relu.dot(relu_z) + b_relu.
        |x-x*|₁ = sum(state_error_s).
        @param lyapunov_lower The lower bound of the Lyapunov function. Set to
        None if you don't want to impose a lower bound.
        @param lyapunov_upper The upper bound of the Lyapunov function. Set to
        None if you don't want to impose an upper bound.
        @parm milp The GurobiTorchMIP instance.
        @param a_relu The ReLU function can be written as
        a_relu.dot(relu_z) + b_relu. a_relu, b_relu, relu_z are returned from
        add_relu_output_constraint().
        @param V_lambda λ.
        @param relu_x_equilibrium ReLU(x*)
        @param state_error_s The slack variable returned from
        add_state_error_l1_constraint()
        """
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        if lyapunov_lower is not None:
            milp.addLConstr(
                [a_relu, V_lambda * torch.ones((self.system.x_dim,),
                                               dtype=self.system.dtype)],
                [relu_z, state_error_s], sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=lyapunov_lower - b_relu + relu_x_equilibrium)
        if lyapunov_upper is not None:
            milp.addLConstr(
                [a_relu, V_lambda * torch.ones((self.system.x_dim,),
                                               dtype=self.system.dtype)],
                [relu_z, state_error_s], sense=gurobipy.GRB.LESS_EQUAL,
                rhs=lyapunov_upper - b_relu + relu_x_equilibrium)


class LyapunovDiscreteTimeHybridSystem(LyapunovHybridLinearSystem):
    """
    For a discrete time autonomous hybrid linear system
    x[n+1] = Aᵢ*x[n] + cᵢ
    if Pᵢ * x[n] <= qᵢ
    i = 1, ..., K.
    we want to learn a ReLU network as the Lyapunov function for the system.
    The condition for the Lyapunov function is that
    V(x*) = 0
    V(x) > 0 ∀ x ≠ x*
    V(x[n+1]) - V(x[n]) ≤ -ε * V(x[n])
    where x* is the equilibrium point.
    We will first formulate this condition as the optimal cost of a certain
    mixed-integer linear program (MILP) being non-positive. The optimal cost
    is the loss function of our neural network. We will compute the gradient of
    this loss function w.r.t to network weights/bias, and then call gradient
    based optimization (SGD/Adam) to reduce the loss.
    """

    def __init__(self, system, lyapunov_relu):
        """
        @param system A AutonomousHybridLinearSystem instance.
        """
        super(LyapunovDiscreteTimeHybridSystem, self).__init__(
            system, lyapunov_relu)

    def lyapunov_derivative(
            self, x, x_equilibrium, V_lambda, epsilon):
        """
        Compute the Lyapunov derivative condition
        V(x[n+1]) - V(x[n]) + εV(x[n])
        Note that there might be multiple posible x[n+1] for a given x[n]
        (when x[n] is on the boundary of two neighbouring modes), so we return
        a list of values as all possible V(x[n+1]) - V(x[n]) + εV(x[n])
        @param x The current state x[n].
        @return V_derivative_possible A list of possible
        V(x[n+1]) - V(x[n]) + εV(x[n])
        """
        assert(isinstance(x, torch.Tensor))
        assert(x.shape == (self.system.x_dim,))
        x_next_possible = self.system.possible_dx(x)
        relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        V_next_possible = [self.lyapunov_value(
            x_next, x_equilibrium, V_lambda, relu_at_equilibrium)
            for x_next in x_next_possible]
        V = self.lyapunov_value(
            x, x_equilibrium, V_lambda, relu_at_equilibrium)
        return [V_next - V + epsilon * V for V_next in V_next_possible]

    def lyapunov_derivative_as_milp(
        self, x_equilibrium, V_lambda, epsilon, eps_type: ConvergenceEps, *,
            lyapunov_lower=None, lyapunov_upper=None, x_warmstart=None):
        """
        We assume that the Lyapunov function
        V(x) = ReLU(x) - ReLU(x*) + λ|x-x*|₁, where x* is the equilibrium
        state.
        In order to prove that the system converges exponentially, with a
        convergence rate between [ε_min, ε_max], we need to show
        ε_min <= -(V(x[n+1]) - V(x[n])) / V(x[n]) <= ε_max. To show this, we
        consider the following two MILPs
        MILP1, for converge rate lower bound
        max (ε_min-1)*V(x[n]) + V(x[n+1])
        s.t lower <= V(x[n]) <= upper

        MILP2
        max -V(x[n+1]) + (1-ε_max)*V(x[n])
        s.t lower <= V(x[n]) <= upper

        In order to prove that the system converges asymptotically (but not
        necessarily exponentially), we only need to prove that dV is strictly
        negative. We choose to prove that
        V(x[n+1]) - V(x[n]) ≤ −ε |x[n] − x*|₁, we could check the optimality
        of an MILP to determine the asympotic convergence.
        MILP3
        max  V(x[n+1]) - V(x[n]) + ε |x[n] − x*|₁
        s.t lower <= V(x[n]) <= upper
        @param V_lambda λ in the documentation above.
        @param eps_type The interpretation of epsilon. If eps_type=ExpLower,
        then we formulate MILP1. If eps_type=ExpUpper, then we formulate
        MILP2. If eps_type=Asymp, then we formulate MILP3.
        @param lyapunov_lower the "lower" bound in the documentation above. If
        lyapunov_lower = None, then we ignore the lower bound on V(x[n]).
        @param lyapunov_upper the "upper" bound in the documentation above. If
        lyapunov_upper = None, then we ignore the upper bound on V(x[n]).
        @param epsilon Depending on eps_type, epsilon has different
        interpretations. It could be used to verify either exponential or
        asymptotic convergence.
        @param x_warmstart tensor of size self.system.x_dim. If provided, will
        use x_warmstart as initial guess for the *binary* variables of the
        milp. Instead of warm start beta with the binary variable solution from
        the previous iteration, we choose to recompute beta using the previous
        adversarial state `x` in the current neural network, so as to make
        sure that this initial guess of beta is always a feasible solution.
        @return (milp, x, x_next, s, gamma, z, z_next, beta, beta_next)
        where milp is a GurobiTorchMILP object.
        The decision variables of the MILP are
        (x[n], beta[n], gamma[n], x[n+1], s[n], z[n], z[n+1], beta[n+1])
        """
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (self.system.x_dim,))
        if lyapunov_lower is not None:
            assert(isinstance(lyapunov_lower, float))
        if lyapunov_upper is not None:
            assert(isinstance(lyapunov_upper, float))
        assert(isinstance(V_lambda, float))
        assert(isinstance(epsilon, float))
        assert(isinstance(eps_type, ConvergenceEps))

        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)

        # x_next is the variable x[n+1]
        x_next = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x[n+1]")
        # create the decision variables
        x = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x")

        # x is the variable x[n]
        s, gamma = self.add_system_constraint(milp, x, x_next)
        # warmstart the binary variables
        if x_warmstart is not None and (isinstance(
                    self.system, relu_system.AutonomousReLUSystem)
                or isinstance(
                    self.system,
                    relu_system.AutonomousReLUSystemGivenEquilibrium)
                or isinstance(
                    self.system,
                    relu_system.AutonomousResidualReLUSystemGivenEquilibrium)):
            relu_to_optimization.set_activation_warmstart(
                self.system.dynamics_relu, gamma, x_warmstart)

        # Add the mixed-integer constraint that formulates the output of
        # ReLU(x[n]).
        # z is the slack variable to write the output of ReLU(x[n]) with mixed
        # integer linear constraints.
        (z, beta, a_out, b_out) = self.add_relu_output_constraint(milp, x)

        # warmstart the binary variables
        if x_warmstart is not None:
            relu_to_optimization.set_activation_warmstart(
                self.lyapunov_relu, beta, x_warmstart)

        # Now compute ReLU(x*)
        relu_x_equilibrium = self.lyapunov_relu.forward(x_equilibrium)

        # Now add the mixed-integer linear constraint to represent
        # |x[n] - x*|₁. To do so, we introduce the slack variable
        # s_x_norm, beta_x_norm.
        # s_x_norm(i) = |x[n](i) - x*(i)|
        (s_x_norm, beta_x_norm) = self.add_state_error_l1_constraint(
            milp, x_equilibrium, x, slack_name="|x[n]-x*|",
            binary_var_name="beta_x_norm")
        # Now add the mixed-integer linear constraint to represent
        # |x[n+1] - x*|₁. To do so, we introduce the slack variable
        # s_x_next_norm, beta_x_next_norm.
        # s_x_next_norm(i) = |x[n+1](i) - x*(i)|
        (s_x_next_norm, beta_x_next_norm) = self.add_state_error_l1_constraint(
            milp, x_equilibrium, x_next, slack_name="|x[n+1]-x*|",
            binary_var_name="beta_x_next_norm")

        # Now add the constraint
        # lower <= ReLU(x[n]) - ReLU(x*) + λ|x[n]-x*|₁ <= upper
        self.add_lyapunov_bounds_constraint(
            lyapunov_lower, lyapunov_upper, milp, a_out, b_out, V_lambda,
            relu_x_equilibrium, z, s_x_norm)

        # Now write the ReLU output ReLU(x[n+1]) as mixed integer linear
        # constraints
        (z_next, beta_next, _, _) = self.add_relu_output_constraint(
            milp, x_next)

        # warmstart the binary variables
        if x_warmstart is not None:
            relu_to_optimization.set_activation_warmstart(
                self.lyapunov_relu, beta_next,
                self.system.step_forward(x_warmstart))

        # For MILP1, the cost function is (ε-1)*V(x[n]) + V(x[n+1]), equals to
        # max ReLU(x[n+1]) + λ|x[n+1]-x*|₁ - ReLU(x[n]) - λ|x[n]-x*|₁ +
        #     ε * (ReLU(x[n]) - ReLU(x*) + λ|x[n]-x*|₁)
        # For MILP2, the cost function is the negation of MILP1.
        if eps_type == ConvergenceEps.ExpLower or \
                eps_type == ConvergenceEps.ExpUpper:
            obj_coeff = [a_out, (epsilon - 1) * a_out, V_lambda * torch.ones(
                (self.system.x_dim,), dtype=self.system.dtype),
                (epsilon - 1) * V_lambda * torch.ones(
                    (self.system.x_dim,), dtype=self.system.dtype)]
            obj_constant = epsilon * (b_out - relu_x_equilibrium.squeeze())
            obj_vars = [z_next, z, s_x_next_norm, s_x_norm]
            if eps_type == ConvergenceEps.ExpLower:
                milp.setObjective(
                    obj_coeff, obj_vars, obj_constant, gurobipy.GRB.MAXIMIZE)
            elif eps_type == ConvergenceEps.ExpUpper:
                milp.setObjective(
                    [-c for c in obj_coeff], obj_vars, -obj_constant,
                    gurobipy.GRB.MAXIMIZE)
        elif eps_type == ConvergenceEps.Asymp:
            # For asymptotic convergence, the cost is
            # V(x[n+1]) - V(x[n]) + ε |x[n] − x*|₁
            # = ReLU(x[n+1]) + λ|x[n+1]-x*|₁ - ReLU(x[n]) + (ε-λ)|x[n]-x*|₁
            milp.setObjective([a_out, -a_out, V_lambda * torch.ones(
                (self.system.x_dim,), dtype=self.system.dtype),
                (epsilon - V_lambda) * torch.ones(
                    (self.system.x_dim,), dtype=self.system.dtype)],
                [z_next, z, s_x_next_norm, s_x_norm], 0.,
                gurobipy.GRB.MAXIMIZE)
        else:
            raise Exception("unknown eps_type")
        return (milp, x, beta, gamma, x_next, s, z, z_next, beta_next)

    def lyapunov_derivative_loss_at_samples(
        self, V_lambda, epsilon, state_samples, x_equilibrium, eps_type,
            margin=0.):
        """
        We will sample states x̅ⁱ, i=1,...N, and we would like the Lyapunov
        function to decrease on these sampled states x̅ⁱ. We denote l(x) as the
        function we want to penalize, and define a loss as
        mean(max(l(x̅ⁱ) + margin, 0))
        Depending on eps_type, l is defined as
        1. If we want to prove the exponential convergence rate is larger than
           epsilon, then l(x) = V(x_next) - V(x) + ε*V(x)
        2. If we want to prove the exponential convergence rate is smaller
           than epsilon, then l(x) = -(V(x_next) - V(x) + ε*V(x))
        3. If we want to prove the asymptotic convergence, then
           l(x) = V(x_next) - V(x) + ε*|x−x*|₁
        @param V_lambda λ in the Lyapunov function.
        @param epsilon ε in the Lyapunov function.
        @param state_samples The sampled state x̅[n], state_samples[i] is the
        i'th sample x̅ⁱ[n]
        @param x_equilibrium x*.
        @param eps_type The interpretation of epsilon. Whether we prove
        exponential or asymptotic convergence.
        @param margin We might want to shift the margin for the Lyapunov
        loss.
        @return loss The loss
        mean(max(V(x̅ⁱ[n+1]) - V(x̅ⁱ[n]) + ε*V(x̅ⁱ[n]) + margin, 0))
        """
        assert(isinstance(V_lambda, float))
        assert(isinstance(epsilon, float))
        assert(isinstance(state_samples, torch.Tensor))
        assert(state_samples.shape[1] == self.system.x_dim)
        assert(isinstance(eps_type, ConvergenceEps))
        # TODO(hongkai.dai): write this as a batched operation for relu
        # systems.
        state_next = torch.stack([
            self.system.step_forward(state_samples[i]) for i in
            range(state_samples.shape[0])], dim=0)

        return self.lyapunov_derivative_loss_at_samples_and_next_states(
            V_lambda, epsilon, state_samples, state_next,
            x_equilibrium, eps_type, margin)

    def lyapunov_derivative_loss_at_samples_and_next_states(
            self, V_lambda, epsilon, state_samples, state_next,
            x_equilibrium, eps_type, margin=0.):
        """
        We will sample states x̅ⁱ, i=1,...N, and we would like the Lyapunov
        function to decrease on these sampled states x̅ⁱ. We denote l(x) as the
        function we want to penalize, and define a loss as
        mean(max(l(x̅ⁱ) + margin, 0))
        Depending on eps_type, l is defined as
        1. If we want to prove the exponential convergence rate is larger than
           epsilon, then l(x) = V(x_next) - V(x) + ε*V(x)
        2. If we want to prove the exponential convergence rate is smaller
           than epsilon, then l(x) = -(V(x_next) - V(x) + ε*V(x))
        3. If we want to prove the asymptotic convergence, then
           l(x) = V(x_next) - V(x) + ε*|x−x*|₁
        The lyapunov function is
        ReLU(x) - ReLU(x*) + λ|x-x*|₁
        @param V_lambda λ in the Lyapunov function.
        @param epsilon ε in the Lyapunov function.
        @param state_samples The sampled state x̅[n], state_samples[i] is the
        i'th sample x̅ⁱ[n]
        @param state_next The next state x̅[n+1], state_next[i] is the next
        state for the i'th sample x̅ⁱ[n+1]
        @param x_equilibrium x*.
        @param margin We might want to shift the margin for the Lyapunov
        loss.
        @return loss The loss
        mean(max(V(x̅ⁱ[n+1]) - V(x̅ⁱ[n]) + ε*V(x̅ⁱ[n]) + margin, 0))
        """
        assert(isinstance(V_lambda, float))
        assert(isinstance(epsilon, float))
        assert(isinstance(state_samples, torch.Tensor))
        assert(state_samples.shape[1] == self.system.x_dim)
        assert(isinstance(state_next, torch.Tensor))
        assert(state_next.shape[1] == self.system.x_dim)
        assert(state_samples.shape[0] == state_next.shape[0])
        assert(isinstance(eps_type, ConvergenceEps))
        relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        v1 = self.lyapunov_value(
            state_samples, x_equilibrium, V_lambda, relu_at_equilibrium)
        v2 = self.lyapunov_value(
            state_next, x_equilibrium, V_lambda, relu_at_equilibrium)
        if eps_type == ConvergenceEps.ExpLower:
            return torch.nn.HingeEmbeddingLoss(margin=margin)(
                -(v2 - v1 + epsilon * v1),
                torch.tensor(-1.).to(state_samples.device))
        elif eps_type == ConvergenceEps.ExpUpper:
            return torch.nn.HingeEmbeddingLoss(margin=margin)(
                (v2 - v1 + epsilon * v1),
                torch.tensor(-1.).to(state_samples.device))
        elif eps_type == ConvergenceEps.Asymp:
            return torch.nn.HingeEmbeddingLoss(margin=margin)(
                -(v2 - v1 + epsilon * torch.norm(
                    state_samples - x_equilibrium, p=1, dim=1)),
                torch.tensor(-1.).to(state_samples.device))
        else:
            raise Exception("Unknown eps_type")


class LyapunovContinuousTimeHybridSystem(LyapunovHybridLinearSystem):
    """
    For a continuous time autonomous hybrid linear system
    ẋ = Aᵢx + gᵢ if Pᵢx ≤ qᵢ
    we want to learn a ReLU network as the Lyapunov function for the system.
    The condition for the Lyapunov function is that
    V(x*) = 0
    V(x) > 0 ∀ x ≠ x*
    V̇(x) ≤ -ε V(x)
    This proves that the system converges exponentially to x*.
    We will formulate these conditions as the optimal objective of certain
    mixed-integer linear programs (MILP) being non-positive/non-negative.
    """

    def __init__(self, system, lyapunov_relu):
        super(LyapunovContinuousTimeHybridSystem, self).__init__(
            system, lyapunov_relu)

    def lyapunov_derivative(
            self, x, x_equilibrium, V_lambda, epsilon):
        """
        Compute V̇(x) + εV(x) for a given x.
        Notice that V̇(x) can have multiple values for a given x, for two
        reasons:
        1. When the input to a (leaky) ReLU unit is exactly 0, the gradient of
           the ReLU output w.r.t ReLU unit input can be either 1 or 0.
        2. When the state is at the boundary of two hybrid modes, ẋ could take
           two values.
        This function return a list of all possible values.
        @param x The state to be evaluated at.
        @param x_equilbrium The equilibrium state.
        @param V_lambda λ in defining Lyapunov as
        V(x)=ReLU(x) - ReLU(x*)+λ|x-x*|₁
        @param epsilon ε in V̇(x) + εV(x)
        @return possible_lyapunov_derivatives A list of torch tensors
        representing all possible V̇(x) + εV(x).
        """
        assert(isinstance(x, torch.Tensor))
        assert(x.shape == (self.system.x_dim,))
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (self.system.x_dim,))
        assert(isinstance(V_lambda, float))
        assert(isinstance(epsilon, float))
        V = self.lyapunov_value(x, x_equilibrium, V_lambda)
        xdot_all = self.system.possible_dx(x)
        possible_activation_patterns = relu_to_optimization.\
            compute_all_relu_activation_patterns(self.lyapunov_relu, x)
        dReLU_dx_all = [relu_to_optimization.ReLUGivenActivationPattern(
            self.lyapunov_relu, self.system.x_dim, pattern,
            self.system.dtype)[0] for
            pattern in possible_activation_patterns]
        # ∂|x-x*|₁/∂x can have different values if x(i) = x*(i).
        state_error_grad_queue = queue.Queue()
        state_error_grad_queue.put([])
        for i in range(self.system.x_dim):
            state_error_grad_queue_len = state_error_grad_queue.qsize()
            for _ in range(state_error_grad_queue_len):
                queue_front = state_error_grad_queue.get()
                if x[i] > x_equilibrium[i]:
                    queue_front_clone = queue_front.copy()
                    queue_front_clone.append(1.)
                    state_error_grad_queue.put(queue_front_clone)
                elif x[i] < x_equilibrium[i]:
                    queue_front_clone = queue_front.copy()
                    queue_front_clone.append(-1.)
                    state_error_grad_queue.put(queue_front_clone)
                else:
                    queue_front_clone = queue_front.copy()
                    queue_front_clone.append(1.)
                    state_error_grad_queue.put(queue_front_clone)
                    queue_front_clone = queue_front.copy()
                    queue_front_clone.append(-1.)
                    state_error_grad_queue.put(queue_front_clone)
        state_error_grad = [None] * state_error_grad_queue.qsize()
        for i in range(len(state_error_grad)):
            state_error_grad[i] = torch.tensor(
                state_error_grad_queue.get(), dtype=self.system.dtype)

        # First compute dV/dx, then compute Vdot = dV/dx * xdot
        dV_dx = [None] * (len(dReLU_dx_all) * len(state_error_grad))
        for i in range(len(dReLU_dx_all)):
            for j in range(len(state_error_grad)):
                dV_dx[i * len(state_error_grad) + j] = \
                    dReLU_dx_all[i].squeeze() + \
                    V_lambda * state_error_grad[j].squeeze()
        Vdot_all = [None] * (len(dV_dx) * len(xdot_all))
        for i in range(len(dV_dx)):
            for j in range(len(xdot_all)):
                Vdot_all[i * len(xdot_all) + j] = dV_dx[i] @ xdot_all[j]
        return [Vdot + epsilon * V for Vdot in Vdot_all]

    def __compute_Aisi_bounds(self):
        """
        Compute the element-wise bounds on Aᵢsᵢ
        Aᵢsᵢ = Aᵢx when the mode i is active. Otherwise Aᵢsᵢ= 0
        return (Aisi_lower, Aisi_upper) Aisi_lower[i]/Aisi_upper[i] is the
        lower/upper bound of Aᵢsᵢ.
        """
        Aisi_lower = [None] * self.system.num_modes
        Aisi_upper = [None] * self.system.num_modes
        for i in range(self.system.num_modes):
            Aix_lower, Aix_upper = self.system.mode_derivative_bounds(i)
            Aisi_lower[i] = torch.min(
                torch.from_numpy(Aix_lower),
                torch.zeros(self.system.x_dim, dtype=self.system.dtype))
            Aisi_upper[i] = torch.max(
                torch.from_numpy(Aix_upper),
                torch.zeros(self.system.x_dim, dtype=self.system.dtype))
        return (Aisi_lower, Aisi_upper)

    def __compute_gigammai_bounds(self):
        """
        Compute the element-wise bounds on gᵢγᵢ
        return (gigammai_lower, gigammai_upper)
        gigammai_lower[i]/gigammai_upper[i] is the lower/upper bound of gᵢγᵢ.
        """
        gigammai_lower = [None] * self.system.num_modes
        gigammai_upper = [None] * self.system.num_modes
        for i in range(self.system.num_modes):
            gigammai_lower[i] = torch.min(
                torch.zeros(self.system.x_dim, dtype=self.system.dtype),
                self.system.g[i])
            gigammai_upper[i] = torch.max(
                torch.zeros(self.system.x_dim, dtype=self.system.dtype),
                self.system.g[i])
        return (gigammai_lower, gigammai_upper)

    def add_relu_gradient_times_Aisi(
            self, milp, s, beta,
            Aisi_lower=None, Aisi_upper=None, slack_name="z"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add sum_i ∂ReLU(x)/∂x*Aᵢsᵢ as mixed-integer linear constraints.
        @param s The slack variable to write the hybrid linear dynamics as
        mixed-integer linear constraint. Returned from add_system_constraint()
        @param beta The binary variable to determine the activation of the
        (leaky) ReLU units in the network, returned from
        add_relu_output_constraint()
        @param Aisi_lower Aisi_lower[i] is the lower bound of Aᵢsᵢ
        @param Aisi_upper Aisi_upper[i] is the lower bound of Aᵢsᵢ
        @return (z, a_out) z and a_out are both lists. z[i] are the slack
        variables to write ∂ReLU(x)/∂x*Aᵢsᵢ as mixed-integer linear constraint
        a_out[i].dot(z[i]) = ∂ReLU(x)/∂x*Aᵢsᵢ
        """
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        assert(isinstance(s, list))
        assert(isinstance(beta, list))

        z = [None] * self.system.num_modes
        a_out = [None] * self.system.num_modes
        if (Aisi_lower is None or Aisi_upper is None):
            Aisi_lower, Aisi_upper = self.__compute_Aisi_bounds()
        else:
            assert(len(Aisi_lower) == self.system.num_modes)
            assert(len(Aisi_upper) == self.system.num_modes)
        for i in range(self.system.num_modes):
            # First write ∂ReLU(x)/∂x*Aᵢsᵢ
            a_out[i], A_Aisi, A_z, A_beta, rhs, _, _ = \
                self.lyapunov_relu_free_pattern.output_gradient_times_vector(
                    Aisi_lower[i], Aisi_upper[i])
            z[i] = milp.addVars(
                A_z.shape[1], lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name=slack_name+"["+str(i)+"]")
            A_si = A_Aisi @ self.system.A[i]
            milp.addMConstrs(
                [A_si, A_z, A_beta],
                [s[i*self.system.x_dim:(i+1)*self.system.x_dim], z[i], beta],
                sense=gurobipy.GRB.LESS_EQUAL, b=rhs,
                name="milp_relu_gradient_times_Aisi")
        return (z, a_out)

    def add_relu_gradient_times_xdot(
            self, milp, xdot, beta,
            xdot_lower, xdot_upper, slack_name="z"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add sum_i ∂ReLU(x)/∂x*ẋ as mixed-integer linear constraints.
        @param xdot The variable representing xdot
        @param beta The binary variable to determine the activation of the
        (leaky) ReLU units in the network, returned from
        add_relu_output_constraint()
        @param xdot_lower xdot_lower[i] is the lower bound of ẋ
        @param xdot_upper xdot_upper[i] is the lower bound of ẋ
        @return (z, a_out) z and a_out are both lists. z[i] are the slack
        variables to write ∂ReLU(x)/∂x*ẋ as mixed-integer linear constraint
        a_out[i].dot(z[i]) = ∂ReLU(x)/∂x*ẋ
        """
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        assert(isinstance(beta, list))

        assert(isinstance(xdot_lower, np.ndarray))
        assert(isinstance(xdot_upper, np.ndarray))
        assert(xdot_lower.shape == (self.system.x_dim,))
        assert(xdot_upper.shape == (self.system.x_dim,))
        # First write ∂ReLU(x)/∂x*ẋ
        a_out, A_xdot, A_z, A_beta, rhs, _, _ = \
            self.lyapunov_relu_free_pattern.output_gradient_times_vector(
                torch.from_numpy(xdot_lower),
                torch.from_numpy(xdot_upper))
        z = milp.addVars(
            A_z.shape[1], lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name=slack_name)
        milp.addMConstrs(
            [A_xdot, A_z, A_beta], [xdot, z, beta],
            sense=gurobipy.GRB.LESS_EQUAL, b=rhs,
            name="milp_relu_gradient_times_xdot")
        return (z, a_out)

    def add_relu_gradient_times_gigammai(
        self, milp, gamma, beta,
            gigammai_lower=None, gigammai_upper=None, slack_name="z"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add sum_i ∂ReLU(x)/∂x *gᵢγᵢ as mixed-integer linear constraints.
        @param gamma The binary variable indicating the active mode in the
        hybrid dynamical system. Returned from add_system_constraint()
        @param beta The binary variable to determine the activation of the
        (leaky) ReLU units in the network, returned from
        add_relu_output_constraint()
        @param gigammai_lower gigammai_lower[i] is the lower bound of gᵢγᵢ
        @param gigammai_upper gigammai_upper[i] is the upper bound of gᵢγᵢ
        @return (z, a_out) z and a_out are both lists. z[i] are the slack
        variables to write ∂ReLU(x)/∂x*gᵢγᵢ as mixed-integer linear constraint
        a_out[i].dot(z[i]) = ∂ReLU(x)/∂x*gᵢγᵢ
        """
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        assert(isinstance(gamma, list))
        assert(len(gamma) == self.system.num_modes)
        assert(isinstance(beta, list))
        z = [None] * self.system.num_modes
        a_out = [None] * self.system.num_modes
        if gigammai_lower is None or gigammai_upper is None:
            gigammai_lower = [None] * self.system.num_modes
            gigammai_upper = [None] * self.system.num_modes
            for i in range(self.system.num_modes):
                gigammai_lower[i] = torch.min(
                    torch.zeros(self.system.x_dim, dtype=self.system.dtype),
                    self.system.g[i])
                gigammai_upper[i] = torch.max(
                    torch.zeros(self.system.x_dim, dtype=self.system.dtype),
                    self.system.g[i])
        for i in range(self.system.num_modes):
            a_out[i], A_gigammai, A_z, A_beta, rhs, _, _ =\
                self.lyapunov_relu_free_pattern.output_gradient_times_vector(
                    gigammai_lower[i], gigammai_upper[i])
            z[i] = milp.addVars(
                A_z.shape[1], lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name=slack_name)
            A_gammai = A_gigammai @ self.system.g[i]
            milp.addMConstrs(
                [A_gammai.reshape((-1, 1)), A_z, A_beta],
                [[gamma[i]], z[i], beta], sense=gurobipy.GRB.LESS_EQUAL,
                b=rhs, name="milp_relu_gradient_times_gigammai")
        return (z, a_out)

    def add_sign_state_error_times_Aisi(
            self, milp, s, alpha, Aisi_lower=None, Aisi_upper=None,
            slack_name="z"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Adds ∑ᵢ ∑ⱼ sign(x(j)-x*(j))*(Aᵢsᵢ)(j) as mixed-integer linear
        constraints.
        @param s The slack variable representing x in each mode. This is
        returned from add_system_constraint().
        @param alpha Binary variables. α(i)=1 => x(i)≥x*(i),
        α(i)=0 => x(i)≤x*(i). This is returned from
        add_state_error_l1_constraint()
        @return (z, z_coeff, s_coeff). z is the continuous slack variable in
        the mixed integer linear constraints. z[i][j] = α(j) *(Aᵢsᵢ)(j),
        z_coeff[i].dot(z[i]) + s_coeff[i].dot(sᵢ) = sign(x(i)-x*(i))*Aᵢsᵢ
        """
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        assert(isinstance(s, list))
        assert(len(s) == self.system.x_dim * self.system.num_modes)
        assert(isinstance(alpha, list))
        assert(len(alpha) == self.system.x_dim)
        if Aisi_lower is None or Aisi_upper is None:
            Aisi_lower, Aisi_upper = self.__compute_Aisi_bounds()
        else:
            assert(isinstance(Aisi_lower, list))
            assert(isinstance(Aisi_upper, list))
            assert(len(Aisi_lower) == self.system.num_modes)
            assert(len(Aisi_upper) == self.system.num_modes)
        z = [None] * self.system.num_modes
        z_coeff = [None] * self.system.num_modes
        s_coeff = [None] * self.system.num_modes
        for i in range(self.system.num_modes):
            # since sign(x(j) - x*(j)) = 2 * α(j) - 1
            # sign(x(j)-x*(j)) * (Aᵢsᵢ)(j) = 2α(j)*(Aᵢsᵢ)(j) - (Aᵢsᵢ)(j)
            z[i] = milp.addVars(
                self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name=slack_name)
            z_coeff[i] = \
                2 * torch.ones(self.system.x_dim, dtype=self.system.dtype)
            s_coeff[i] = -torch.sum(self.system.A[i], dim=0)
            for j in range(self.system.x_dim):
                Ain_Aisi, Ain_z, Ain_alpha, rhs_in = utils.\
                    replace_binary_continuous_product(
                        Aisi_lower[i][j], Aisi_upper[i][j])
                Ain_si = Ain_Aisi.reshape((-1, 1)) @ \
                    self.system.A[i][j].reshape((1, -1))
                milp.addMConstrs(
                    [Ain_si, Ain_z.reshape((-1, 1)),
                     Ain_alpha.reshape((-1, 1))],
                    [s[i*self.system.x_dim:(i+1)*self.system.x_dim],
                     [z[i][j]], [alpha[j]]], sense=gurobipy.GRB.LESS_EQUAL,
                    b=rhs_in)
        return (z, z_coeff, s_coeff)

    def add_sign_state_error_times_gigammai(
            self, milp, gamma, alpha, gigammai_lower=None, gigammai_upper=None,
            slack_name="z"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Adds ∑ᵢ ∑ⱼ sign(x(j)-x*(j))*(gᵢγᵢ)(j) as mixed-integer linear
        constraints.
        @param gamma The binary variable representing the active mode. This is
        returned from add_system_constraint().
        @param alpha Binary variables. α(i)=1 => x(i)≥x*(i),
        α(i)=0 => x(i)≤x*(i). This is returned from
        add_state_error_l1_constraint()
        @param gigammai_lower The lower bound of gᵢγᵢ, this is returned from
        __compute_gigammai_bounds()
        @param gigammai_upper The upper bound of gᵢγᵢ, this is returned from
        __compute_gigammai_bounds()
        @return (z, z_coeff, gamma_coeff). z is the continuous slack variable
        in the mixed integer linear constraints. z[i][j] = α(j) *(gᵢγᵢ)(j),
        z_coeff[i].dot(z[i]) + gamma_coeff[i].dot(γᵢ) = sign(x(i)-x*(i))*gᵢγᵢ
        """
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        assert(isinstance(gamma, list))
        assert(len(gamma) == self.system.num_modes)
        assert(isinstance(alpha, list))
        assert(len(alpha) == self.system.x_dim)
        if gigammai_lower is None or gigammai_upper is None:
            gigammai_lower, gigammai_upper = self.__compute_gigammai_bounds()
        else:
            assert(isinstance(gigammai_lower, list))
            assert(isinstance(gigammai_upper, list))
            assert(len(gigammai_lower) == self.system.num_modes)
            assert(len(gigammai_upper) == self.system.num_modes)
        z = [None] * self.system.num_modes
        z_coeff = [None] * self.system.num_modes
        gamma_coeff = [None] * self.system.num_modes
        for i in range(self.system.num_modes):
            z[i] = milp.addVars(
                self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name=slack_name+"["+str(i)+"]")
            # ∑ⱼ sign(x(j)-x*(j))*(gᵢγᵢ)(j)
            # = ∑ⱼ 2α(j)*(gᵢγᵢ)(j) - (gᵢγᵢ)(j)
            z_coeff[i] = 2 * torch.ones(
                self.system.x_dim, dtype=self.system.dtype)
            gamma_coeff[i] = -torch.sum(self.system.g[i]).unsqueeze(0)
            for j in range(self.system.x_dim):
                Ain_gigammai, Ain_z, Ain_alpha, rhs = utils.\
                    replace_binary_continuous_product(
                        gigammai_lower[i][j], gigammai_upper[i][j])
                Ain_gammai = Ain_gigammai * self.system.g[i][j]
                milp.addMConstrs(
                    [Ain_gammai.reshape((-1, 1)), Ain_z.reshape((-1, 1)),
                     Ain_alpha.reshape((-1, 1))],
                    [[gamma[i]], [z[i][j]], [alpha[j]]],
                    sense=gurobipy.GRB.LESS_EQUAL, b=rhs)
        return (z, z_coeff, gamma_coeff)

    def add_sign_state_error_times_xdot(
            self, milp, xdot, alpha, xdot_lower, xdot_upper, slack_name="z"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Adds ∑ᵢ sign(x(i)-x*(i))* ẋ(i) as mixed-integer linear constraints.
        @param xdot The decision variable representing ẋ
        @param alpha Binary variables. α(i)=1 => x(i)≥x*(i),
        α(i)=0 => x(i)≤x*(i). This is returned from
        add_state_error_l1_constraint()
        @return (z, z_coeff, xdot_coeff). z is the continuous slack variable
        in the mixed integer linear constraints. z[i] = α(i) *ẋ(i),
        z_coeff.dot(z) + xdot_coeff[i].dot(xdot) = sign(x(i)-x*(i))*ẋ
        """
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        assert(isinstance(xdot, list))
        assert(len(xdot) == self.system.x_dim)
        assert(isinstance(alpha, list))
        assert(len(alpha) == self.system.x_dim)
        assert(isinstance(xdot_lower, np.ndarray))
        assert(isinstance(xdot_upper, np.ndarray))
        # since sign(x(i) - x*(i)) = 2 * α(i) - 1
        # sign(x(i)-x*(i)) * xdot(j) = 2α(i)*xdot(i) - xdot(i)
        z = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name=slack_name)
        z_coeff = \
            2 * torch.ones(self.system.x_dim, dtype=self.system.dtype)
        xdot_coeff = -torch.ones(
            self.system.x_dim, dtype=self.system.dtype)
        for i in range(self.system.x_dim):
            Ain_xdot, Ain_z, Ain_alpha, rhs_in = utils.\
                replace_binary_continuous_product(
                    xdot_lower[i], xdot_upper[i])
            milp.addMConstrs(
                [Ain_xdot.reshape((-1, 1)), Ain_z.reshape((-1, 1)),
                 Ain_alpha.reshape((-1, 1))],
                [[xdot[i]], [z[i]], [alpha[i]]],
                sense=gurobipy.GRB.LESS_EQUAL, b=rhs_in)
        return (z, z_coeff, xdot_coeff)

    def lyapunov_derivative_as_milp2(
            self, x_equilibrium, V_lambda, epsilon, eps_type,
            lyapunov_lower=None, lyapunov_upper=None):
        """
        We assume that the Lyapunov function
        V(x) = ReLU(x) - ReLU(x*) + λ|x-x*|₁, where x* is the equilibrium
        state.
        Formulate the Lyapunov condition
        V̇(x) ≤ -ε V(x) for all x satisfying
        lower <= V(x) <= upper
        as the maximal of following optimization problem is no larger
        than 0.
        max V̇(x) + ε * V(x)
        s.t lower <= V(x) <= upper
        We would formulate this optimization problem as an MILP.

        @param x_equilibrium The equilibrium state.
        @param V_lambda λ in the documentation above.
        @param epsilon The exponential convergence rate.
        @param lyapunov_lower the "lower" bound in the documentation above. If
        lyapunov_lower = None, then we ignore the lower bound on V(x).
        @param lyapunov_upper the "upper" bound in the documentation above. If
        lyapunov_upper = None, then we ignore the upper bound on V(x).
        @param epsilon The rate of exponential convergence. If the goal is to
        verify convergence but not exponential convergence, then set epsilon
        to 0.
        @return (milp, x, relu_beta, gamma) milp is the GurobiTorchMILP
        object such that if the maximal of this MILP is 0, the condition
        V̇(x) ≤ -ε V(x) is satisfied. x is the decision variable in the milp
        as the adversarial state (the state with the maximal violation of
        Lyapunov condition V̇(x) ≤ -ε V(x), and relu_beta is the binary
        variable representing the activation pattern of the ReLU network.
        gamma is the binary variable representing the active hybrid mode
        for the adversarial state x.
        """
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (self.system.x_dim,))
        if lyapunov_lower is not None:
            assert(isinstance(lyapunov_lower, float))
        if lyapunov_upper is not None:
            assert(isinstance(lyapunov_upper, float))
        assert(isinstance(V_lambda, float))
        assert(isinstance(epsilon, float))
        assert(isinstance(eps_type, ConvergenceEps))
        if eps_type != ConvergenceEps.ExpLower:
            raise NotImplementedError

        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)

        x = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x")
        s, gamma = self.add_system_constraint(milp, x, None)

        # V̇ = ∂V/∂x(∑ᵢ Aᵢsᵢ + gᵢγᵢ)
        #   = ∑ᵢ(∂ReLU(x)/∂x*Aᵢsᵢ + ∂ReLU(x)/∂x*gᵢγᵢ
        #       + λ*sign(x-x*) * Aᵢsᵢ + λ*sign(x-x*)*gᵢγᵢ)
        # In order to compute ∂ReLU(x)/∂x*Aᵢsᵢ, ∂ReLU(x)/∂x*gᵢγᵢ, we first
        # introduce the binary variable β, which represents the activation of
        # each (leaky) ReLU unit in the network. Then we can call
        # output_gradient_times_vector().

        # We first get the mixed-integer linear constraint, which encode the
        # activation of beta and the network input.
        (relu_z, relu_beta, a_relu_out, b_relu_out) = \
            self.add_relu_output_constraint(milp, x)

        # for each mode, we want to compute ∂V/∂x*Aᵢsᵢ, ∂V/∂x*gᵢγᵢ.
        # where ∂V/∂x=∂ReLU(x)/∂x + λ*sign(x-x*)
        # In order to handle the part on λ*sign(x-x*) * Aᵢsᵢ, λ*sign(x-x*)*gᵢγᵢ
        # we first introduce binary variable α, such that
        # α(j) = 1 => x(j) - x*(j) >= 0
        # α(j) = 0 => x(j) - x*(j) <= 0
        # Hence sign(x(j) - x*(j)) = 2 * α - 1
        # t[i] = |x(i)-x*(i)|
        (t, alpha) = self.add_state_error_l1_constraint(
                milp, x_equilibrium, x, slack_name="t",
                binary_var_name="alpha")

        # Now add the constraint
        # lower <= ReLU(x[n]) - ReLU(x*) + λ|x[n]-x*|₁ <= upper
        relu_x_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        self.add_lyapunov_bounds_constraint(
            lyapunov_lower, lyapunov_upper, milp, a_relu_out, b_relu_out,
            V_lambda, relu_x_equilibrium, relu_z, t)

        # z1[i] is the slack variable to write ∂ReLU(x)/∂x*Aᵢsᵢ as
        # mixed-integer linear constraints. cost_z1_coef is the coefficient of
        # z1 in the objective.
        Aisi_lower, Aisi_upper = self.__compute_Aisi_bounds()
        gigammai_lower, gigammai_upper = self.__compute_gigammai_bounds()
        z1, cost_z1_coef = self.add_relu_gradient_times_Aisi(
            milp, s, relu_beta, Aisi_lower,
            Aisi_upper, slack_name="z1")
        z2, cost_z2_coef = self.add_relu_gradient_times_gigammai(
            milp, gamma, relu_beta, slack_name="z2")
        # z3[i] is the slack variable to write sign(x-x*)*Aᵢsᵢ as mixed-integer
        # linear constraints.
        z3, z3_coef, s_coef = self.add_sign_state_error_times_Aisi(
            milp, s, alpha, Aisi_lower, Aisi_upper, slack_name="z3")
        # cost_z3_coef[i] is the coefficient of z3[i] in the cost function.
        cost_z3_coef = [coef * V_lambda for coef in z3_coef]
        cost_s_coef = [coef * V_lambda for coef in s_coef]
        # z4[i] is the slack variable to write sign(x-x*)*gᵢγᵢ as mixed-integer
        # linear constraints.
        z4, z4_coef, gamma_coef = self.add_sign_state_error_times_gigammai(
            milp, gamma, alpha, gigammai_lower, gigammai_upper,
            slack_name="z4")
        # cost_z4_coef[i] is the coefficient of z4[i] in the cost function.
        cost_z4_coef = [coef * V_lambda for coef in z3_coef]
        # cost_gamma_coef[i] is the coefficient of gamma[i] in the cost
        # function.
        cost_gamma_coef = [coef * V_lambda for coef in gamma_coef]

        # The cost is
        # max V̇ + εV
        #   = ∑ᵢ∂V/∂x(Aᵢsᵢ + gᵢγᵢ) + ε(ReLU(x) - ReLU(x*) + λ|x-x*|₁)
        # We know that ∂V/∂x = ∂ReLU(x)/∂x + ρ*sign(x-x*) and
        # ∂ReLU(x)/∂x * Aᵢsᵢ = z1_coef[i].dot(z1)
        # ∂ReLU(x)/∂x * gᵢγᵢ = z2_coef[i].dot(z2)
        # ρ*sign(x-x*) * Aᵢsᵢ = z3_coeff[i].dot(z3) * s_coef[i].dot(sᵢ)
        # ρ*sign(x-x*) * gᵢγᵢ = z4_coeff[i].dot(z4) * gamma_coef[i].dot(γᵢ)
        # ReLU(x) = a_relu_out.dot(relu_z) + b_relu_out
        # λ|x-x*|₁ = λ * sum(t)
        cost_vars = [mode_var for var_list in [z1, z2, z3, z4] for mode_var
                     in var_list]
        cost_coeffs = [mode_coef for coef_list in [
            cost_z1_coef, cost_z2_coef, cost_z3_coef, cost_z4_coef] for
            mode_coef in coef_list]
        cost_coeffs.append(torch.cat(cost_s_coef))
        cost_vars.append(s)
        cost_coeffs.append(torch.cat(cost_gamma_coef))
        cost_vars.append(gamma)

        cost_vars.append(relu_z)
        cost_coeffs.append(a_relu_out * epsilon)

        cost_vars.append(t)
        cost_coeffs.append(
            epsilon * V_lambda * torch.ones(
                self.system.x_dim, dtype=self.system.dtype))
        milp.setObjective(
            cost_coeffs, cost_vars,
            epsilon * b_relu_out - epsilon * relu_x_equilibrium.squeeze(),
            gurobipy.GRB.MAXIMIZE)

        return (milp, x, relu_beta, gamma)

    def lyapunov_derivative_as_milp(
            self, x_equilibrium, V_lambda, epsilon, eps_type: ConvergenceEps,
            *, lyapunov_lower=None, lyapunov_upper=None):
        """
        We assume that the Lyapunov function
        V(x) = ReLU(x) - ReLU(x*) + λ|x-x*|₁, where x* is the equilibrium
        state.
        Formulate the Lyapunov condition
        V̇(x) ≤ -ε V(x) for all x satisfying
        lower <= V(x) <= upper
        as the maximal of following optimization problem is no larger
        than 0.
        max V̇(x) + ε * V(x)
        s.t lower <= V(x) <= upper
        We would formulate this optimization problem as an MILP.
        This is an alternative formulation different from
        lyapunov_derivative_as_milp()

        @param x_equilibrium The equilibrium state.
        @param V_lambda λ in the documentation above.
        @param epsilon The exponential convergence rate.
        @param lyapunov_lower the "lower" bound in the documentation above. If
        lyapunov_lower = None, then we ignore the lower bound on V(x).
        @param lyapunov_upper the "upper" bound in the documentation above. If
        lyapunov_upper = None, then we ignore the upper bound on V(x).
        @param epsilon The rate of exponential convergence. If the goal is to
        verify convergence but not exponential convergence, then set epsilon
        to 0.
        @return (milp, x, relu_beta, gamma) milp is the GurobiTorchMILP
        object such that if the maximal of this MILP is 0, the condition
        V̇(x) ≤ -ε V(x) is satisfied. x is the decision variable in the milp
        as the adversarial state (the state with the maximal violation of
        Lyapunov condition V̇(x) ≤ -ε V(x), and relu_beta is the binary
        variable representing the activation pattern of the ReLU network.
        gamma is the binary variable representing the active hybrid mode
        for the adversarial state x.
        """
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (self.system.x_dim,))
        if lyapunov_lower is not None:
            assert(isinstance(lyapunov_lower, float))
        if lyapunov_upper is not None:
            assert(isinstance(lyapunov_upper, float))
        assert(isinstance(V_lambda, float))
        assert(isinstance(epsilon, float))
        assert(isinstance(eps_type, ConvergenceEps))
        if eps_type != ConvergenceEps.ExpLower:
            raise NotImplementedError

        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)

        x = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x")
        xdot = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="xdot")
        s, gamma = self.add_system_constraint(milp, x, xdot)

        # V̇ = ∂V/∂x * ẋ
        #   = ∂ReLU(x)/∂x*ẋ + λ*sign(x-x*) *ẋ
        # In order to compute ∂ReLU(x)/∂x*ẋ, we first
        # introduce the binary variable β, which represents the activation of
        # each (leaky) ReLU unit in the network. Then we can call
        # output_gradient_times_vector().

        # We first get the mixed-integer linear constraint, which encode the
        # activation of beta and the network input.
        (relu_z, relu_beta, a_relu_out, b_relu_out) = \
            self.add_relu_output_constraint(milp, x)

        # for each mode, we want to compute ∂V/∂x*ẋ
        # where ∂V/∂x=∂ReLU(x)/∂x + λ*sign(x-x*)
        # In order to handle the part on λ*sign(x-x*) * ẋ
        # we first introduce binary variable α, such that
        # α(j) = 1 => x(j) - x*(j) >= 0
        # α(j) = 0 => x(j) - x*(j) <= 0
        # Hence sign(x(j) - x*(j)) = 2 * α - 1
        # t[i] = |x(i)-x*(i)|
        (t, alpha) = self.add_state_error_l1_constraint(
                milp, x_equilibrium, x, slack_name="t",
                binary_var_name="alpha")

        # Now add the constraint
        # lower <= ReLU(x[n]) - ReLU(x*) + λ|x[n]-x*|₁ <= upper
        relu_x_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        self.add_lyapunov_bounds_constraint(
            lyapunov_lower, lyapunov_upper, milp, a_relu_out, b_relu_out,
            V_lambda, relu_x_equilibrium, relu_z, t)

        # z1 is the slack variable to write ∂ReLU(x)/∂x*ẋ as
        # mixed-integer linear constraints. cost_z1_coef is the coefficient of
        # z1 in the objective.
        xdot_lower = self.system.dx_lower
        xdot_upper = self.system.dx_upper
        z1, cost_z1_coef = self.add_relu_gradient_times_xdot(
            milp, xdot, relu_beta, xdot_lower,
            xdot_upper, slack_name="z1")
        # z2 is the slack variable to write sign(x-x*)*ẋ as mixed-integer
        # linear constraints.
        z2, z2_coef, xdot_coef = self.add_sign_state_error_times_xdot(
            milp, xdot, alpha, xdot_lower, xdot_upper, slack_name="z2")
        # cost_z3_coef[i] is the coefficient of z3[i] in the cost function.
        cost_z2_coef = z2_coef * V_lambda
        cost_xdot_coef = xdot_coef * V_lambda

        # The cost is
        # max V̇ + εV
        #   = ∑ᵢ∂V/∂x * ẋᵢ + ε(ReLU(x) - ReLU(x*) + λ|x-x*|₁)
        # We know that ∂V/∂x = ∂ReLU(x)/∂x + λ*sign(x-x*) and
        # ∂ReLU(x)/∂x * ẋ = z1_coef.dot(z1)
        # ρ*sign(x-x*) * ẋ = cost_z2_coeff.dot(z2) * cost_xdot_coef.dot(xdot)
        # ReLU(x) = a_relu_out.dot(relu_z) + b_relu_out
        # λ|x-x*|₁ = λ * sum(t)
        cost_vars = [z1, z2, xdot]
        cost_coeffs = [cost_z1_coef, cost_z2_coef, cost_xdot_coef]

        cost_vars.append(relu_z)
        cost_coeffs.append(a_relu_out * epsilon)

        cost_vars.append(t)
        cost_coeffs.append(
            epsilon * V_lambda * torch.ones(
                self.system.x_dim, dtype=self.system.dtype))
        milp.setObjective(
            cost_coeffs, cost_vars,
            epsilon * b_relu_out - epsilon * relu_x_equilibrium.squeeze(),
            gurobipy.GRB.MAXIMIZE)

        return (milp, x, relu_beta, gamma)

    def lyapunov_derivative_loss_at_samples(
        self, V_lambda, epsilon, state_samples, x_equilibrium, eps_type,
            margin=0.):
        """
        We will sample states x̅ⁱ, i=1,...N, and we would like the Lyapunov
        function to decrease on these sampled states x̅ⁱ. We denote l(x) as the
        function we want to penalize, and define a loss as
        mean(max(l(x̅ⁱ) + margin, 0))
        Depending on eps_type, l is defined as
        1. If we want to prove the exponential convergence rate is larger than
           epsilon, then l(x) = V̇(x) + ε*V(x)
        2. If we want to prove the exponential convergence rate is smaller
           than epsilon, then l(x) = -(V̇(x) + ε*V(x))
        3. If we want to prove the asymptotic convergence, then
           l(x) = V̇(x) + ε*|x−x*|₁
        The lyapunov function is
        ReLU(x) - ReLU(x*) + λ|x-x*|₁
        @param V_lambda ρ in the Lyapunov function.
        @param state_samples The sampled state x̅. state_samples[i] is the i'th
        sampled state x̅ⁱ
        @param x_equilibrium x*.
        @param margin We might want to shift the margin for the Lyapunov
        loss.
        @return loss The loss mean(max(V̇(x̅ⁱ) + ε*V(x̅ⁱ) + margin, 0))
        """
        assert(isinstance(V_lambda, float))
        assert(isinstance(epsilon, float))
        assert(isinstance(state_samples, torch.Tensor))
        assert(state_samples.shape[1] == self.system.x_dim)
        assert(isinstance(eps_type, ConvergenceEps))
        xdot = torch.empty(
            (state_samples.shape[0], self.system.x_dim),
            dtype=self.system.dtype)
        for i in range(state_samples.shape[0]):
            # First compute the next state dx̅/dt
            mode = self.system.mode(state_samples[i])
            if mode is None:
                raise Exception(
                    "lyapunov_derivative_loss_at_samples: the input " +
                    f"state_sample {state_samples[i]} is not in any mode of " +
                    "the hybrid system.")
            xdot[i] = self.system.step_forward(state_samples[i], mode)

        return self.lyapunov_derivative_loss_at_samples_and_next_states(
            V_lambda, epsilon, state_samples, xdot, x_equilibrium,
            eps_type, margin)

    def lyapunov_derivative_loss_at_samples_and_next_states(
            self, V_lambda, epsilon, state_samples, xdot_samples,
            x_equilibrium, eps_type, margin=0.):
        """
        We will sample states x̅ⁱ, i=1,...N, and we would like the Lyapunov
        function to decrease on these sampled states x̅ⁱ. We denote l(x) as the
        function we want to penalize, and define a loss as
        mean(max(l(x̅ⁱ) + margin, 0))
        Depending on eps_type, l is defined as
        1. If we want to prove the exponential convergence rate is larger than
           epsilon, then l(x) = V̇(x) + ε*V(x)
        2. If we want to prove the exponential convergence rate is smaller
           than epsilon, then l(x) = -(V̇(x) + ε*V(x))
        3. If we want to prove the asymptotic convergence, then
           l(x) = V̇(x) + ε*|x−x*|₁
        The lyapunov function is
        ReLU(x) - ReLU(x*) + λ|x-x*|₁
        @param V_lambda λ in the Lyapunov function.
        @param state_samples The sampled state x̅. state_samples[i] is the i'th
        sampled state x̅ⁱ
        @param xdot_samples The state derivative dx̅/dt
        @param x_equilibrium x*.
        @param margin We might want to shift the margin for the Lyapunov
        loss.
        @return loss The loss mean(max(V̇(x̅ⁱ) + ε*V(x̅ⁱ) + margin, 0))
        """
        assert(isinstance(V_lambda, float))
        assert(isinstance(epsilon, float))
        assert(isinstance(state_samples, torch.Tensor))
        assert(state_samples.shape[1] == self.system.x_dim)
        assert(isinstance(xdot_samples, torch.Tensor))
        assert(xdot_samples.shape[1] == self.system.x_dim)
        assert(state_samples.shape[0] == xdot_samples.shape[0])
        assert(isinstance(eps_type, ConvergenceEps))

        num_samples = state_samples.shape[0]
        # First compute ∂V/∂x using pytorch autodiff.
        dReLU_dx = [None] * num_samples
        for i in range(num_samples):
            # TODO(hongkai.dai): figure out how to remove this for loop.
            x_var = torch.autograd.Variable(
                state_samples[i], requires_grad=True)
            relu_output = self.lyapunov_relu(x_var)
            dReLU_dx[i] = torch.autograd.grad(
                relu_output, x_var, create_graph=True, allow_unused=True)[0]
        dReLU_dx_tensor = torch.stack(dReLU_dx)
        dV_dx = dReLU_dx_tensor + \
            V_lambda * torch.sign(state_samples - x_equilibrium)
        Vdot = torch.sum(dV_dx * xdot_samples, dim=1)
        V = self.lyapunov_value(
            state_samples, x_equilibrium, V_lambda).squeeze()
        if eps_type == ConvergenceEps.ExpLower:
            val = Vdot + epsilon * V
        elif eps_type == ConvergenceEps.ExpUpper:
            val = -(Vdot + epsilon * V)
        elif eps_type == ConvergenceEps.Asymp:
            val = (Vdot + epsilon * torch.norm(
                state_samples - x_equilibrium, p=1, dim=1))
        else:
            raise Exception("Unknown eps_type")
        return torch.nn.HingeEmbeddingLoss(margin=margin)(
            -val, torch.tensor(-1))
