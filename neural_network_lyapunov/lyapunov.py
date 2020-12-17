# -*- coding: utf-8 -*-
import gurobipy
import torch

from enum import Enum

import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils


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
        function. The Lyapunov function is ReLU(x) - ReLU(x*) + λ|R*(x-x*)|₁
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
                 torch.from_numpy(self.system.x_up_all),
                 mip_utils.PropagateBoundsMethod.IA)
        relu_z, relu_beta = milp.add_mixed_integer_linear_constraints(
            mip_constr_return, x, None, slack_name, binary_var_name,
            "milp_relu_ineq", "milp_relu_eq", "")
        return (relu_z, relu_beta, mip_constr_return.Aout_slack.squeeze(),
                mip_constr_return.Cout.squeeze())

    def add_state_error_l1_constraint(
            self, milp, x_equilibrium, x, *, R=None, slack_name="s",
            binary_var_name="alpha", fixed_R=True):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add the L1 loss |R*(x-x*)|₁ as mixed-integer linear constraints.
        return (s, alpha) s is the continuous slack variable,
        s(i) = |R[i, :] * (x - x*)|, alpha is the binary variable,
        alpha(i) = 1 => s(i) >= 0, alpha(i) = 0 => s(i) <= 0
        @param R A matrix. We want this matrix to have full column rank. If
        R=None, then we use identity as R.
        @param fixed_R If set to False, then we will treat R as free
        variables, which eventually we will need to compute the gradient of R.
        Hence if fixed_R is True, then we compute the range of R*(x-x*)
        by linear programming (LP), otherwise we compute the range by interval
        arithmetic (IA).
        """
        if not torch.all(torch.from_numpy(self.system.x_lo_all) <=
                         x_equilibrium) or\
                not torch.all(torch.from_numpy(self.system.x_up_all) >=
                              x_equilibrium):
            raise Exception("add_state_error_l1_constraint: we currently " +
                            "require that x_lo <= x_equilibrium <= x_up")
        R = _get_R(R, self.system.x_dim, x_equilibrium.device)
        # R should have full column rank, so that s = 0 implies x = x*
        assert(R.shape[0] >= self.system.x_dim)
        s_dim = R.shape[0]
        s = milp.addVars(
            s_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name=slack_name)
        alpha = milp.addVars(
            s_dim, vtype=gurobipy.GRB.BINARY, name=binary_var_name)

        if fixed_R:
            s_lb, s_ub = mip_utils.compute_range_by_lp(
                R.detach().numpy(), (-R @ x_equilibrium).detach().numpy(),
                self.system.x_lo_all, self.system.x_up_all, None, None)
        else:
            s_lb, s_ub = mip_utils.compute_range_by_IA(
                R, -R @ x_equilibrium, torch.from_numpy(self.system.x_lo_all),
                torch.from_numpy(self.system.x_up_all))
        for i in range(s_dim):
            if s_lb[i] < 0 and s_ub[i] > 0:
                # Add the constraint s[i] = |R[i, :] * (x - x*)|
                # We first convert the absolute value to mixed-integer linear
                # constraint, and then add the constraint
                # Ain_x * (R[i, :] * (x - x*)) + Ain_s * s + Ain_alpha * alpha
                # <= rhs_in
                Ain_x, Ain_s, Ain_alpha, rhs_in = utils.\
                    replace_absolute_value_with_mixed_integer_constraint(
                        s_lb[i], s_ub[i], dtype=torch.float64)
                rhs = rhs_in + (Ain_x.reshape((-1, 1)) @ R[i].reshape((1, -1))
                                @ x_equilibrium.reshape((-1, 1))).reshape((-1))
                milp.addMConstrs([
                    Ain_x.reshape((-1, 1)) @ R[i].reshape((1, -1)),
                    Ain_s.reshape((-1, 1)), Ain_alpha.reshape((-1, 1))],
                    [x, [s[i]], [alpha[i]]], sense=gurobipy.GRB.LESS_EQUAL,
                    b=rhs)
            elif s_lb[i] >= 0:
                # Add the constraint s[i] = R[i, :] * (x - x*)
                milp.addLConstr([
                    torch.tensor([1], dtype=torch.float64), -R[i]],
                    [[s[i]], x], sense=gurobipy.GRB.EQUAL,
                    rhs=-R[i] @ x_equilibrium)
                # Add the constraint alpha[i] = 1
                milp.addLConstr(
                    [torch.tensor([1], dtype=torch.float64)], [[alpha[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=1.)
            else:
                # Add the constraint s[i] = -R[i, :] * (x - x*)
                milp.addLConstr([
                    torch.tensor([1], dtype=torch.float64), R[i]],
                    [[s[i]], x], sense=gurobipy.GRB.EQUAL,
                    rhs=R[i] @ x_equilibrium)
                # Add the constraint alpha[i] = 0
                milp.addLConstr(
                    [torch.tensor([1], dtype=torch.float64)], [[alpha[i]]],
                    sense=gurobipy.GRB.EQUAL, rhs=0.)

        return (s, alpha)

    def lyapunov_value(
        self, x, x_equilibrium, V_lambda, *, R=None,
            relu_at_equilibrium=None):
        """
        Compute the value of the Lyapunov function as
        ReLU(x) - ReLU(x*) + λ|R*(x-x*)|₁
        where |R*(x-x*)|₁ is the 1-norm of R*(x-x*).
        @param x a torch tensor. Evaluate Lyapunov function at this point.
        @param x_equilibrium a torch tensor. The equilibrium state x*.
        @param V_lambda λ in the documentation above.
        @param R R in the documentation above. It should be a full column rank
        matrix. If R=None, then we use identity as R.
        @param relu_at_equilibrium. ReLU(x*). If set to None, then we compute
        ReLU(x*) in this function.
        """
        R = _get_R(R, self.system.x_dim, x_equilibrium.device)
        if relu_at_equilibrium is None:
            relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        if x.shape == (self.system.x_dim,):
            # A single state.
            return self.lyapunov_relu.forward(x) - relu_at_equilibrium +\
                V_lambda * torch.norm(R @ (x - x_equilibrium), p=1)
        else:
            # A batch of states.
            assert(x.shape[1] == self.system.x_dim)
            return self.lyapunov_relu(x).squeeze() - relu_at_equilibrium + \
                V_lambda * torch.norm(R @ (x - x_equilibrium).T, p=1, dim=0)

    def lyapunov_positivity_as_milp(
        self, x_equilibrium, V_lambda, V_epsilon, *, R, fixed_R,
            x_warmstart=None):
        """
        For a ReLU network, in order to determine if the function
        V(x) = ReLU(x) - ReLU(x*) + λ * |R * (x - x*)|₁
        where |R*(x - x*)|₁ is the 1-norm of the vector R*(x - x*).
        satisfies the positivity constraint of Lyapunov condition
        V(x) > 0 ∀ x ≠ x*
        We check a strong condition
        V(x) ≥ ε |R*(x - x*)|₁ ∀ x
        where ε is a small positive number.To check if the stronger condition
        is satisfied, we can solve the following optimization problem
        max x  (ε-λ) * |R*(x - x*)|₁ - ReLU(x) + ReLU(x*)
        We can formulate this optimization problem as a mixed integer linear
        program, solve the for optimal solution of this program. If the optimal
        cost is no larger than 0, then we proved the positivity constraint
        V(x) > 0 ∀ x ≠ x*
        @param x_equilibrium The equilibrium state x*.
        @param V_lambda ρ in the documentation above.
        @param V_epsilon A scalar. ε in the documentation above.
        @param R This matrix must have full column rank, we will use the
        1-norm of R * (x - x*). If R=None, then we use identity matrix as R.
        @param fixed_R Whether R is fixed or not. If R is fixed, we compute
        the range of R * (x-x*) by LP, otherwise we compute it through
        interval arithmetics.
        @param x_warmstart tensor of size self.system.x_dim. If provided, will
        use x_warmstart as initial guess for the *binary* variables of the
        milp. Instead of warm start beta with the binary variable solution from
        the previous iteration, we choose to recompute beta using the previous
        adversarial state `x` in the current neural network, so as to make
        sure that this initial guess of beta is always a feasible solution.
        @return (milp, x) milp is a GurobiTorchMILP instance, x is the decision
        variable for state.
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

        # Now write the 1-norm |R*(x - x*)|₁ as mixed-integer linear
        # constraints.
        (s, gamma) = self.add_state_error_l1_constraint(
            milp, x_equilibrium, x, R=R, slack_name="s",
            binary_var_name="gamma", fixed_R=fixed_R)

        milp.setObjective(
            [-a_out.squeeze(),
             (V_epsilon-V_lambda) *
             torch.ones((len(s),), dtype=dtype)],
            [z, s], constant=-b_out + relu_x_equilibrium.squeeze(),
            sense=gurobipy.GRB.MAXIMIZE)
        return (milp, x)

    def lyapunov_positivity_loss_at_samples(
            self, relu_at_equilibrium, x_equilibrium,
            state_samples, V_lambda, epsilon, *, R, margin=0.):
        """
        We will sample a state xⁱ, and we would like the Lyapunov function to
        be larger than 0 at xⁱ. Hence we define the loss as
        mean(max(-V(xⁱ) + ε |R * (xⁱ - x*)|₁ + margin, 0))
        @param relu_at_equilibrium A 0-D tensor. ReLU(x*)
        @param x_equilibrium x* in the documentation above.
        @param state_samples A batch of sampled states, state_samples[i] is
        the i'th sample xⁱ.
        @param V_lambda λ in the documentation above.
        @param epsilon ε in the documentation above.
        @param R Should be a full column rank matrix. We use the 1-norm of
        R * (xⁱ - x*)
        @param margin The margin used in the hinge loss.
        """
        assert(isinstance(relu_at_equilibrium, torch.Tensor))
        assert(isinstance(state_samples, torch.Tensor))
        assert(state_samples.shape[1] == self.system.x_dim)
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (self.system.x_dim,))
        assert(isinstance(V_lambda, float))
        assert(isinstance(margin, float))
        R = _get_R(R, self.system.x_dim, state_samples.device)
        return torch.nn.HingeEmbeddingLoss(margin=margin)(
            self.lyapunov_value(
                state_samples, x_equilibrium, V_lambda, R=R,
                relu_at_equilibrium=relu_at_equilibrium) -
            epsilon * torch.norm(
                R @ (state_samples - x_equilibrium).T, p=1, dim=0),
            torch.tensor(-1.).to(state_samples.device))

    def add_lyapunov_bounds_constraint(
        self, lyapunov_lower, lyapunov_upper, milp, a_relu, b_relu, V_lambda,
            relu_x_equilibrium, relu_z, state_error_s):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add constraint lower <= V(x) <= upper to @p milp, where the Lyapunov
        function V(x) = ReLU(x) - ReLU(x*) + λ|R*(x-x*)|₁.
        Also we have ReLU(x) = a_relu.dot(relu_z) + b_relu.
        |R(x-x*)|₁ = sum(state_error_s).
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
        s_dim = len(state_error_s)
        if lyapunov_lower is not None:
            milp.addLConstr(
                [a_relu, V_lambda * torch.ones((s_dim,),
                                               dtype=self.system.dtype)],
                [relu_z, state_error_s], sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=lyapunov_lower - b_relu + relu_x_equilibrium)
        if lyapunov_upper is not None:
            milp.addLConstr(
                [a_relu, V_lambda * torch.ones((s_dim,),
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
            self, x, x_equilibrium, V_lambda, epsilon, *, R):
        """
        Compute the Lyapunov derivative condition
        V(x[n+1]) - V(x[n]) + εV(x[n])
        where the Lyapunov function is
        V(x) = ϕ(x) − ϕ(x*) + λ*|R *(x−x*)|₁
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
            x_next, x_equilibrium, V_lambda, R=R,
            relu_at_equilibrium=relu_at_equilibrium)
            for x_next in x_next_possible]
        V = self.lyapunov_value(
            x, x_equilibrium, V_lambda, R=R,
            relu_at_equilibrium=relu_at_equilibrium)
        return [V_next - V + epsilon * V for V_next in V_next_possible]

    def lyapunov_derivative_as_milp(
        self, x_equilibrium, V_lambda, epsilon, eps_type: ConvergenceEps, *,
        R, fixed_R, lyapunov_lower=None, lyapunov_upper=None,
            x_warmstart=None):
        """
        We assume that the Lyapunov function
        V(x) = ReLU(x) - ReLU(x*) + λ|R*(x-x*)|₁, where x* is the equilibrium
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
        V(x[n+1]) - V(x[n]) ≤ −ε |R*(x[n] − x*)|₁, we could check the
        optimality of an MILP to determine the asympotic convergence.
        MILP3
        max  V(x[n+1]) - V(x[n]) + ε |R*(x[n] − x*)|₁
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
        R = _get_R(R, self.system.x_dim, x_equilibrium.device)

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
        # |R*(x[n] - x)*|₁. To do so, we introduce the slack variable
        # s_x_norm, beta_x_norm.
        # s_x_norm(i) = |R[i,:] * (x[n] - x*)|
        (s_x_norm, beta_x_norm) = self.add_state_error_l1_constraint(
            milp, x_equilibrium, x, R=R, slack_name="|x[n]-x*|",
            binary_var_name="beta_x_norm", fixed_R=fixed_R)
        # Now add the mixed-integer linear constraint to represent
        # |R*(x[n+1] - x*)|₁. To do so, we introduce the slack variable
        # s_x_next_norm, beta_x_next_norm.
        # s_x_next_norm(i) = |R[i, :] * (x[n+1] - x*)|
        (s_x_next_norm, beta_x_next_norm) = self.add_state_error_l1_constraint(
            milp, x_equilibrium, x_next, R=R, slack_name="|R*(x[n+1]-x*)|",
            binary_var_name="beta_x_next_norm", fixed_R=fixed_R)

        # Now add the constraint
        # lower <= ReLU(x[n]) - ReLU(x*) + λ|R*(x[n]-x*)|₁ <= upper
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

        s_dim = R.shape[0]

        # For MILP1, the cost function is (ε-1)*V(x[n]) + V(x[n+1]), equals to
        # max ReLU(x[n+1]) + λ|R*(x[n+1]-x*)|₁ - ReLU(x[n]) - λ|R*(x[n]-x*)|₁
        #     + ε * (ReLU(x[n]) - ReLU(x*) + λ|x[n]-x*|₁)
        # For MILP2, the cost function is the negation of MILP1.
        if eps_type == ConvergenceEps.ExpLower or \
                eps_type == ConvergenceEps.ExpUpper:
            obj_coeff = [a_out, (epsilon - 1) * a_out, V_lambda * torch.ones(
                (s_dim,), dtype=self.system.dtype),
                (epsilon - 1) * V_lambda * torch.ones(
                    (s_dim,), dtype=self.system.dtype)]
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
            # V(x[n+1]) - V(x[n]) + ε |R*(x[n] − x*)|₁
            # = ReLU(x[n+1]) + λ|R*(x[n+1]-x*)|₁ - ReLU(x[n]) +
            #  (ε-λ)|R*(x[n]-x*)|₁
            milp.setObjective([a_out, -a_out, V_lambda * torch.ones(
                (s_dim,), dtype=self.system.dtype),
                (epsilon - V_lambda) * torch.ones(
                    (s_dim,), dtype=self.system.dtype)],
                [z_next, z, s_x_next_norm, s_x_norm], 0.,
                gurobipy.GRB.MAXIMIZE)
        else:
            raise Exception("unknown eps_type")
        return (milp, x, beta, gamma, x_next, s, z, z_next, beta_next)

    def lyapunov_derivative_loss_at_samples(
        self, V_lambda, epsilon, state_samples, x_equilibrium, eps_type,
            *, R, margin=0.):
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
        R = _get_R(R, self.system.x_dim, state_samples.device)
        state_next = self.system.step_forward(state_samples)

        return self.lyapunov_derivative_loss_at_samples_and_next_states(
            V_lambda, epsilon, state_samples, state_next,
            x_equilibrium, eps_type, R=R, margin=margin)

    def lyapunov_derivative_loss_at_samples_and_next_states(
            self, V_lambda, epsilon, state_samples, state_next,
            x_equilibrium, eps_type, *, R, margin=0.):
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
        @param exp_type The interpretation of epsilon. If exp_type=ExpLower,
        then the loss wrt to the convergence lower bound. If exp_type=ExpUpper,
        then the loss is with respect to the upper bound
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
        R = _get_R(R, self.system.x_dim, state_samples.device)
        relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        v1 = self.lyapunov_value(
            state_samples, x_equilibrium, V_lambda, R=R,
            relu_at_equilibrium=relu_at_equilibrium)
        v2 = self.lyapunov_value(
            state_next, x_equilibrium, V_lambda, R=R,
            relu_at_equilibrium=relu_at_equilibrium)
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
                    R @ (state_samples - x_equilibrium).T, p=1, dim=0)),
                torch.tensor(-1.).to(state_samples.device))
        else:
            raise Exception("Unknown eps_type")


def _get_R(R, x_dim, device):
    """
    Take matrix R used in the 1-norm |R*(x-x*)|₁.
    """
    assert(isinstance(R, torch.Tensor) or R is None)
    if R is None:
        return torch.eye(x_dim, dtype=torch.float64).to(device)
    else:
        assert(R.shape[1] == x_dim)
        return R.to(device)
