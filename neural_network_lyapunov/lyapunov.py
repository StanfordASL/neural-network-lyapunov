# -*- coding: utf-8 -*-
import gurobipy
import torch
import numpy as np

from enum import Enum
import collections

import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.dynamic_system as dynamic_system


class ConvergenceEps(Enum):
    """
    The epsilon constant in the Lyapunov derivative has different
    interpretations, it could represent the exponential convergence rate,
    where ε_min ≤ −V̇/V ≤ ε_max, or it can represent the asymptotic
    convergence, namely V̇ ≤ −ε |R(x−x*)|₁.
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
        assert (isinstance(system,
                           hybrid_linear_system.AutonomousHybridLinearSystem)
                or isinstance(system, relu_system.AutonomousReLUSystem)
                or isinstance(system,
                              relu_system.AutonomousReLUSystemGivenEquilibrium)
                or isinstance(
                    system,
                    relu_system.AutonomousResidualReLUSystemGivenEquilibrium)
                or isinstance(system, feedback_system.FeedbackSystem))
        self.system = system
        self.lyapunov_relu = lyapunov_relu
        self.lyapunov_relu_free_pattern = \
            relu_to_optimization.ReLUFreePattern(
                lyapunov_relu, self.system.dtype)
        self.network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.IA

    def add_lyap_relu_output_constraint(self,
                                        milp,
                                        x,
                                        slack_name="relu_z",
                                        binary_var_name="relu_beta",
                                        *,
                                        binary_var_type=gurobipy.GRB.BINARY):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add the Lyapunov relu output as mixed-integer linear constraint.
        @return (z, beta, a_out, b_out) z is the continuous slack variable.
        beta is the binary variable indicating whether a (leaky) ReLU unit is
        active or not. The output of the network can be written as
        a_out.dot(z) + b_out
        """
        assert (isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        assert (isinstance(x, list))
        mip_constr_return = self.lyapunov_relu_free_pattern.output_constraint(
            torch.from_numpy(self.system.x_lo_all),
            torch.from_numpy(self.system.x_up_all),
            self.network_bound_propagate_method)
        relu_z, relu_beta = milp.add_mixed_integer_linear_constraints(
            mip_constr_return, x, None, slack_name, binary_var_name,
            "milp_relu_ineq", "milp_relu_eq", "", binary_var_type)
        return (relu_z, relu_beta, mip_constr_return.Aout_slack.squeeze(),
                mip_constr_return.Cout.squeeze(), mip_constr_return)

    def add_state_error_l1_constraint(self,
                                      milp,
                                      x_equilibrium,
                                      x,
                                      *,
                                      R=None,
                                      slack_name="s",
                                      binary_var_name="alpha",
                                      binary_var_type=gurobipy.GRB.BINARY,
                                      binary_for_zero_input=False):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        we add the L1 loss |R*(x-x*)|₁ as mixed-integer linear constraints.
        return (s, alpha) s is the continuous slack variable,
        s(i) = |R[i, :] * (x - x*)|, alpha is the binary variable,
        If binary_for_zero_input is False, then
        alpha is the same dimension as s
        alpha(i) = 1 => s(i) >= 0, alpha(i) = 0 => s(i) <= 0
        If binary_for_zero_input is True, then
        alpha is a list, where each entry of alpha[i] contains 3 binary
        variables.
        alpha[i][0] = 1 => s(i) <= 0
        alpha[i][1] = 1 => s(i) = 0
        alpha[i][2] = 1 => s(i) >= 0
        and we also impose the constraint
        alpha[i][0] + alpha[i][1] + alpha[i][2] = 1
        @param R A matrix. We want this matrix to have full column rank. If
        R=None, then we use identity as R.
        """
        if not torch.all(torch.from_numpy(self.system.x_lo_all)
                         <= x_equilibrium) or\
            not torch.all(torch.from_numpy(self.system.x_up_all)
                          >= x_equilibrium):
            raise Exception("add_state_error_l1_constraint: we currently " +
                            "require that x_lo <= x_equilibrium <= x_up")
        R = _get_R(R, self.system.x_dim, x_equilibrium.device)
        # R should have full column rank, so that s = 0 implies x = x*
        assert (R.shape[0] >= R.shape[1])
        s_dim = R.shape[0]
        # The lower and upper bound of R*(x-x*)
        Rx_lb, Rx_ub = mip_utils.compute_range_by_IA(
            R, -R @ x_equilibrium, torch.from_numpy(self.system.x_lo_all),
            torch.from_numpy(self.system.x_up_all))
        s = [None] * s_dim
        alpha = [None] * s_dim
        for i in range(s_dim):
            mip_cnstr = utils.absolute_value_as_mixed_integer_constraint(
                Rx_lb[i], Rx_ub[i], binary_for_zero_input)
            # The constraint in mip_cnstr is
            # Ain_input * (R[i, :] * (x-x*)) + Ain_slack * s[i] +
            # Ain_binary * alpha <= rhs_in
            # Aeq_input * (R[i, :] * (x-x*)) + Aeq_slack * s[i] +
            # Aeq_binary * alpha = rhs_eq
            mip_cnstr.transform_input(R[i, :].reshape(
                (1, -1)), (-R[i, :] @ x_equilibrium).reshape((-1, )))
            s_i, alpha_i = milp.add_mixed_integer_linear_constraints(
                mip_cnstr, x, None, slack_name + f"[{i}]",
                binary_var_name + f"[{i}]", "l1_ineq", "l1_eq", "",
                binary_var_type)
            assert (len(s_i) == 1)
            s[i] = s_i[0]
            if len(alpha_i) == 1:
                alpha[i] = alpha_i[0]
            else:
                alpha[i] = alpha_i
        return (s, alpha)

    def lyapunov_value(self, x, x_equilibrium, V_lambda, *, R=None):
        """
        Compute the value of the Lyapunov function as
        V(x) = ReLU(x) - ReLU(x*) + λ|R*(x-x*)|₁
        where |R*(x-x*)|₁ is the 1-norm of R*(x-x*).
        x* is the equilibrium state.
        @param x a torch tensor. Evaluate Lyapunov function at this point.
        @param x_equilibrium a torch tensor. The equilibrium state x*.
        @param V_lambda λ in the documentation above.
        @param R R in the documentation above. It should be a full column rank
        matrix. If R=None, then we use identity as R.
        """
        R = _get_R(R, self.system.x_dim, x_equilibrium.device)
        relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        if x.shape == (self.system.x_dim, ):
            # A single state.
            return self.lyapunov_relu.forward(x) - relu_at_equilibrium +\
                V_lambda * torch.norm(R @ (
                    x - x_equilibrium), p=1)
        else:
            # A batch of states.
            assert (x.shape[1] == self.system.x_dim)
            return self.lyapunov_relu(x).squeeze() - \
                relu_at_equilibrium.squeeze() + \
                V_lambda * torch.norm(R @ (
                    x - x_equilibrium).T, p=1,
                    dim=0)

    def _lyapunov_value_as_milp(self, mip, x, x_equilibrium, V_lambda,
                                R) -> (list, list, torch.Tensor, list):
        """
        For an MILP, add the constraints such that we can compute
        V(x) = V_coeff * V_vars + V_constant

        Return:
          V_coeff, V_vars, V_constant: V(x) = V_coeff * V_vars + V_constant
          s: s[i] = |R.row(i)*(x-x*)|
        """
        assert (isinstance(mip, gurobi_torch_mip.GurobiTorchMIP))
        assert (isinstance(x, list))
        assert (len(x) == self.system.x_dim)
        # z is the slack variable to write the output of ReLU network as mixed
        # integer constraints.
        z, beta, a_out, b_out, _ = self.add_lyap_relu_output_constraint(mip, x)
        # Now write the 1-norm |R*(x - x*)|₁ as mixed-integer linear
        # constraints.
        s, gamma = self.add_state_error_l1_constraint(mip,
                                                      x_equilibrium,
                                                      x,
                                                      R=R,
                                                      slack_name="s",
                                                      binary_var_name="gamma")
        # Now set the V(x) = ϕ(x) - ϕ(x*) + λ*|R(x−x*)|₁
        # = a_out * z + b_out - ϕ(x*) + λ * s
        relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        V_coeff = [
            a_out.squeeze(0), V_lambda * torch.ones(
                (len(s), ), dtype=self.system.dtype)
        ]
        V_vars = [z, s]
        V_constant = b_out.squeeze() - relu_at_equilibrium.squeeze()
        return V_coeff, V_vars, V_constant, s

    def lyapunov_positivity_as_milp(self,
                                    x_equilibrium,
                                    V_lambda,
                                    V_epsilon,
                                    *,
                                    R,
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
        @param x_warmstart tensor of size self.system.x_dim. If provided, will
        use x_warmstart as initial guess for the *binary* variables of the
        milp. Instead of warm start beta with the binary variable solution from
        the previous iteration, we choose to recompute beta using the previous
        adversarial state `x` in the current neural network, so as to make
        sure that this initial guess of beta is always a feasible solution.
        @return (milp, x) milp is a GurobiTorchMILP instance, x is the decision
        variable for state.
        """
        assert (isinstance(x_equilibrium, torch.Tensor))
        assert (x_equilibrium.shape == (self.system.x_dim, ))
        assert (isinstance(V_lambda, float))
        assert (isinstance(V_epsilon, float))
        if x_warmstart is not None:
            assert (isinstance(x_warmstart, torch.Tensor))
            assert (x_warmstart.shape == (self.system.x_dim, ))

        dtype = self.system.dtype
        milp = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x = milp.addVars(self.system.x_dim,
                         lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="x")
        # z is the slack variable to write the output of ReLU network as mixed
        # integer constraints.
        z, beta, a_out, b_out, _ = self.add_lyap_relu_output_constraint(
            milp, x)

        # warmstart the binary variables
        if x_warmstart is not None:
            relu_to_optimization.set_activation_warmstart(
                self.lyapunov_relu, beta, x_warmstart)

        # Now write the 1-norm |R*(x - x*)|₁ as mixed-integer linear
        # constraints.
        (s,
         gamma) = self.add_state_error_l1_constraint(milp,
                                                     x_equilibrium,
                                                     x,
                                                     R=R,
                                                     slack_name="s",
                                                     binary_var_name="gamma")

        relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        # Now set the objective as -ϕ(x) + ϕ(x*) + (ε-λ)*|R(x−x*)|₁
        # = -a_out * z - b_out +  ϕ(x*) + (ε-λ) * s
        milp.setObjective([
            -a_out.squeeze(), (V_epsilon - V_lambda) * torch.ones(
                (len(s), ), dtype=dtype)
        ], [z, s],
                          constant=-b_out + relu_at_equilibrium.squeeze(),
                          sense=gurobipy.GRB.MAXIMIZE)

        return (milp, x)

    def lyapunov_positivity_loss_at_samples(self,
                                            x_equilibrium,
                                            state_samples,
                                            V_lambda,
                                            epsilon,
                                            *,
                                            R,
                                            margin=0.,
                                            reduction="mean",
                                            weight=None):
        """
        We will sample a state xⁱ, and we would like the Lyapunov function to
        be larger than 0 at xⁱ. Hence we define the loss as
        mean(max(-V(xⁱ) + ε |R * (xⁱ - x*)|₁ + margin, 0))
        @param x_equilibrium x* in the documentation above.
        @param state_samples A batch of sampled states, state_samples[i] is
        the i'th sample xⁱ.
        @param V_lambda λ in the documentation above.
        @param epsilon ε in the documentation above.
        @param R Should be a full column rank matrix. We use the 1-norm of
        R * (xⁱ - x*)
        @param margin The margin used in the hinge loss.
        @param reduction If reduction="mean", we use the mean loss across all
        samples, if reduction="max", we use the max loss among all samples, if
        reduction="4norm", we use the 4-norm on the loss vector for all
        samples.
        @param weight If set to None, then we use uniform weight of 1 for
        every sample. Otherwise weight should be a vector of the same length
        as the number of samples, whereh weight[i] is the weight of
        state_samples[i].
        """
        assert (isinstance(state_samples, torch.Tensor))
        assert (state_samples.shape[1] == self.system.x_dim)
        assert (isinstance(x_equilibrium, torch.Tensor))
        assert (x_equilibrium.shape == (self.system.x_dim, ))
        assert (isinstance(V_lambda, float))
        assert (isinstance(margin, float))
        assert (reduction in {"mean", "max", "4norm"})
        R = _get_R(R, self.system.x_dim, state_samples.device)
        loss = self.lyapunov_value(
            state_samples, x_equilibrium, V_lambda,
            R=R) - epsilon * torch.norm(
                R @ (state_samples - x_equilibrium).T, p=1, dim=0)
        if reduction == "mean":
            if weight is not None:
                assert (weight.shape == (state_samples.shape[0], ))
                return torch.mean(weight * torch.nn.HingeEmbeddingLoss(
                    margin=margin, reduction="none")(loss, torch.tensor(-1).to(
                        state_samples.device)))
            else:
                return torch.nn.HingeEmbeddingLoss(margin=margin)(
                    loss, torch.tensor(-1.).to(state_samples.device))
        elif reduction == "max":
            return torch.max(
                torch.nn.HingeEmbeddingLoss(margin=margin, reduction="none")(
                    loss, torch.tensor(-1.).to(state_samples.device)))
        elif reduction == "4norm":
            return torch.norm(torch.nn.HingeEmbeddingLoss(
                margin=margin,
                reduction="none")(loss,
                                  torch.tensor(-1.).to(state_samples.device)),
                              p=4)

    def add_lyapunov_bounds_constraint(self, lyapunov_lower, lyapunov_upper,
                                       milp, a_relu, b_relu, V_lambda, relu_z,
                                       relu_at_equilibrium, state_error_s):
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
        add_lyap_relu_output_constraint().
        @param V_lambda λ.
        @param relu_at_equilibrium ReLU(x*)
        @param state_error_s The slack variable returned from
        add_state_error_l1_constraint()
        """
        assert (isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        s_dim = len(state_error_s)
        if lyapunov_lower is not None:
            milp.addLConstr([
                a_relu, V_lambda * torch.ones(
                    (s_dim, ), dtype=self.system.dtype)
            ], [relu_z, state_error_s],
                            sense=gurobipy.GRB.GREATER_EQUAL,
                            rhs=lyapunov_lower - b_relu + relu_at_equilibrium)
        if lyapunov_upper is not None:
            milp.addLConstr([
                a_relu, V_lambda * torch.ones(
                    (s_dim, ), dtype=self.system.dtype)
            ], [relu_z, state_error_s],
                            sense=gurobipy.GRB.LESS_EQUAL,
                            rhs=lyapunov_upper - b_relu + relu_at_equilibrium)

    def _construct_milp_for_roa_boundary(self, V_lambda, R, x_equilibrium):
        """
        Construct an MILP to solve the problem
        min V(x)
        s.t x in the boundary of the box.
        """
        dtype = self.system.dtype
        # I could use the original gurobi interface directly, instead of the
        # gurobi_torch_mip interface, as we don't need to compute the gradient
        # of the results.
        milp = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x_lo = torch.from_numpy(self.system.x_lo_all)
        x_up = torch.from_numpy(self.system.x_up_all)
        x = milp.addVars(self.system.x_dim,
                         lb=x_lo,
                         ub=x_up,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="x[n]")

        z, beta, a_out, b_out, _ = self.add_lyap_relu_output_constraint(
            milp, x)

        # Now add the constraint that x is on the boundary of the box
        # x_lo <= x <= x_up
        # We add binary variables ζ_up and ζ_lo, such that
        # ζ_up(i)=1 => x[i] = x_up[i]
        # ζ_lo(i)=1 => x[i] = x_lo[i]
        # Using big-M trick, we get
        # x[i] >= x_lo[i] + (x_up[i]-x_lo[i]) * ζ_up[i]
        # x[i] <= x_up[i] - (x_up[i]-x_lo[i]) * ζ_lo[i]
        zeta_lo = milp.addVars(self.system.x_dim, vtype=gurobipy.GRB.BINARY)
        zeta_up = milp.addVars(self.system.x_dim, vtype=gurobipy.GRB.BINARY)
        milp.addLConstr([
            torch.ones((self.system.x_dim, ), dtype=dtype),
            torch.ones((self.system.x_dim, ), dtype=dtype)
        ], [zeta_lo, zeta_up],
                        rhs=1.,
                        sense=gurobipy.GRB.EQUAL)
        milp.addMConstr([
            torch.eye(self.system.x_dim, dtype=dtype),
            torch.diag(x_lo - x_up)
        ], [x, zeta_up],
                        b=x_lo,
                        sense=gurobipy.GRB.GREATER_EQUAL)
        milp.addMConstr([
            torch.eye(self.system.x_dim, dtype=dtype),
            torch.diag(x_up - x_lo)
        ], [x, zeta_lo],
                        b=x_up,
                        sense=gurobipy.GRB.LESS_EQUAL)

        relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)

        # Now write the 1-norm |R*(x[n] - x*)|₁ as mixed-integer linear
        # constraints.
        (s,
         gamma) = self.add_state_error_l1_constraint(milp,
                                                     x_equilibrium,
                                                     x,
                                                     R=R,
                                                     slack_name="s",
                                                     binary_var_name="gamma")

        # Objective is
        # min V(x[n])
        # = ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁
        # = a_out * z + b_out + λ * s - ϕ(x*)
        milp.setObjective(
            [a_out.squeeze(), V_lambda * torch.ones(
                (len(s), ), dtype=dtype)], [z, s],
            constant=b_out - relu_at_equilibrium.squeeze(),
            sense=gurobipy.GRB.MINIMIZE)
        return milp, x

    def validate_x_equilibrium(self, x_equilibrium: torch.Tensor):
        """
        Validate that x_equilibrium is acceptable.
        x_equilibrium should be within x_lo and x_up
        """
        assert (isinstance(x_equilibrium, torch.Tensor))
        assert (x_equilibrium.shape == (self.system.x_dim, ))
        assert (np.all(x_equilibrium.detach().numpy() <= self.system.x_up_all))
        assert (np.all(x_equilibrium.detach().numpy() >= self.system.x_lo_all))

    def _lyapunov_gradient(self, x, x_equilibrium, V_lambda, R, zero_tol):
        """
        Compute the gradient ∂V/∂x.
        When the gradient is not unique, we return all the left and right
        gradient.
        """
        assert (x.shape == (self.system.x_dim, ))
        dphidx = utils.relu_network_gradient(self.lyapunov_relu,
                                             x,
                                             zero_tol=zero_tol).squeeze(1)
        dl1dx = V_lambda * utils.l1_gradient(R @ (x - x_equilibrium),
                                             zero_tol=zero_tol) @ R
        dVdx = utils.minkowski_sum(dphidx, dl1dx)
        return dVdx

    def _lyapunov_gradient_batch(self, x, x_equilibrium, V_lambda, R,
                                 create_graph):
        """
        Compute the gradient ∂V/∂x.
        This function assumes x is a batch of state. When there are multiple
        possible subgradients, we take the one returned from pytorch autodiff.
        """
        assert (x.shape[1] == self.system.x_dim)
        x_requires_grad = x.requires_grad
        x.requires_grad = True
        V = self.lyapunov_value(x, x_equilibrium, V_lambda, R=R)
        dVdx = torch.autograd.grad(outputs=V,
                                   inputs=x,
                                   grad_outputs=torch.ones_like(V),
                                   create_graph=create_graph)[0]
        x.requires_grad = x_requires_grad
        return dVdx


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
        @param system A dynamical system.
        """
        super(LyapunovDiscreteTimeHybridSystem,
              self).__init__(system, lyapunov_relu)

    def lyapunov_derivative(self, x, x_equilibrium, V_lambda, epsilon, *, R):
        """
        Compute the Lyapunov derivative condition
        V(x[n+1]) - V(x[n]) + εV(x[n])
        where the Lyapunov function is
        V(x) = ϕ(x) − ϕ(x*) + λ*|R *(x−x*)|₁
        Note that there might be multiple posible x[n+1] for a given x[n]
        (when x[n] is on the boundary of two neighbouring modes), so we return
        a list of values as all possible V(x[n+1]) - V(x[n]) + εV(x[n])
        @param x The current state x[n].
        @param x_equilibrium x* in the documentation above.
        @return V_derivative_possible A list of possible
        V(x[n+1]) - V(x[n]) + εV(x[n])
        """
        assert (isinstance(x, torch.Tensor))
        assert (x.shape == (self.system.x_dim, ))
        x_next_possible = self.system.possible_dx(x)
        V_next_possible = [
            self.lyapunov_value(x_next, x_equilibrium, V_lambda, R=R)
            for x_next in x_next_possible
        ]
        V = self.lyapunov_value(x, x_equilibrium, V_lambda, R=R)
        return [V_next - V + epsilon * V for V_next in V_next_possible]

    def lyapunov_derivative_as_milp(self,
                                    x_equilibrium,
                                    V_lambda,
                                    epsilon,
                                    eps_type: ConvergenceEps,
                                    *,
                                    R,
                                    lyapunov_lower=None,
                                    lyapunov_upper=None,
                                    x_warmstart=None,
                                    binary_var_type=gurobipy.GRB.BINARY):
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
        assert (isinstance(x_equilibrium, torch.Tensor))
        assert (x_equilibrium.shape == (self.system.x_dim, ))
        if lyapunov_lower is not None:
            assert (isinstance(lyapunov_lower, float))
        if lyapunov_upper is not None:
            assert (isinstance(lyapunov_upper, float))
        assert (isinstance(V_lambda, float))
        assert (isinstance(epsilon, float))
        assert (isinstance(eps_type, ConvergenceEps))
        R = _get_R(R, self.system.x_dim, x_equilibrium.device)

        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)

        # x_next is the variable x[n+1]
        x_next = milp.addVars(self.system.x_dim,
                              lb=-gurobipy.GRB.INFINITY,
                              vtype=gurobipy.GRB.CONTINUOUS,
                              name="x[n+1]")
        # create the decision variables
        x = milp.addVars(self.system.x_dim,
                         lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="x")

        # x is the variable x[n]
        system_constraint_return = dynamic_system._add_system_constraint(
            self.system, milp, x, x_next, binary_var_type=binary_var_type)
        s = system_constraint_return.slack
        gamma = system_constraint_return.binary
        # warmstart the binary variables
        if x_warmstart is not None and (
                isinstance(self.system, relu_system.AutonomousReLUSystem)
                or isinstance(self.system,
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
        z, beta, a_out, b_out, lyap_relu_x_mip_cnstr_ret = \
            self.add_lyap_relu_output_constraint(
                milp, x, binary_var_type=binary_var_type)

        # warmstart the binary variables
        if x_warmstart is not None:
            relu_to_optimization.set_activation_warmstart(
                self.lyapunov_relu, beta, x_warmstart)

        # Now compute ReLU(x*)
        relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)

        # Now add the mixed-integer linear constraint to represent
        # |R*(x[n] - x*)|₁. To do so, we introduce the slack variable
        # s_x_norm, beta_x_norm.
        # s_x_norm(i) = |R[i,:] * (x[n] - x*)|
        (s_x_norm, beta_x_norm) = self.add_state_error_l1_constraint(
            milp,
            x_equilibrium,
            x,
            R=R,
            slack_name="|x[n]-x*|",
            binary_var_name="beta_x_norm",
            binary_var_type=binary_var_type)
        # Now add the mixed-integer linear constraint to represent
        # |R*(x[n+1] - x*)|₁. To do so, we introduce the slack variable
        # s_x_next_norm, beta_x_next_norm.
        # s_x_next_norm(i) = |R[i, :] * (x[n+1] - x*)|
        (s_x_next_norm, beta_x_next_norm) = self.add_state_error_l1_constraint(
            milp,
            x_equilibrium,
            x_next,
            R=R,
            slack_name="|R*(x[n+1]-x*)|",
            binary_var_name="beta_x_next_norm",
            binary_var_type=binary_var_type)

        # Now add the constraint
        # lower <= ReLU(x[n]) - ReLU(x*) + λ|R*(x[n]-x*)|₁ <= upper
        self.add_lyapunov_bounds_constraint(lyapunov_lower, lyapunov_upper,
                                            milp, a_out, b_out, V_lambda, z,
                                            relu_at_equilibrium, s_x_norm)

        # Now write the ReLU output ReLU(x[n+1]) as mixed integer linear
        # constraints
        z_next, beta_next, _, _, lyap_relu_x_next_mip_cnstr_ret = \
            self.add_lyap_relu_output_constraint(
                milp, x_next, binary_var_type=binary_var_type)

        # warmstart the binary variables
        if x_warmstart is not None:
            relu_to_optimization.set_activation_warmstart(
                self.lyapunov_relu, beta_next,
                self.system.step_forward(x_warmstart))

        s_dim = R.shape[0]

        # For MILP1, the cost function is (ε-1)*V(x[n]) + V(x[n+1]), equals to
        # max ϕ(x[n+1]) + λ|R*(x[n+1]-x*)|₁
        #       + (ε-1) * ϕ(x[n]) − εϕ(x*) + (ε−1)λ|R*(x[n]−x*)|₁
        # For MILP2, the cost function is the negation of MILP1.
        if eps_type == ConvergenceEps.ExpLower or \
                eps_type == ConvergenceEps.ExpUpper:
            obj_coeff = [
                a_out, (epsilon - 1) * a_out,
                V_lambda * torch.ones((s_dim, ), dtype=self.system.dtype),
                (epsilon - 1) * V_lambda * torch.ones(
                    (s_dim, ), dtype=self.system.dtype)
            ]
            obj_constant = epsilon * (b_out - relu_at_equilibrium.squeeze())
            obj_vars = [z_next, z, s_x_next_norm, s_x_norm]
            if eps_type == ConvergenceEps.ExpLower:
                milp.setObjective(obj_coeff, obj_vars, obj_constant,
                                  gurobipy.GRB.MAXIMIZE)
            elif eps_type == ConvergenceEps.ExpUpper:
                milp.setObjective([-c for c in obj_coeff], obj_vars,
                                  -obj_constant, gurobipy.GRB.MAXIMIZE)
        elif eps_type == ConvergenceEps.Asymp:
            # For asymptotic convergence, the cost is
            # V(x[n+1]) - V(x[n]) + ε |R*(x[n] − x*)|₁
            # = ϕ(x[n+1] + λ|R(x[n+1]−x*)|₁−ϕ(x[n]) + (ε-λ)|R*(x[n]-x*)|₁
            milp.setObjective([
                a_out, -a_out, V_lambda * torch.ones(
                    (s_dim, ), dtype=self.system.dtype),
                (epsilon - V_lambda) * torch.ones(
                    (s_dim, ), dtype=self.system.dtype)
            ], [z_next, z, s_x_next_norm, s_x_norm], 0., gurobipy.GRB.MAXIMIZE)
        else:
            raise Exception("unknown eps_type")
        LyapDerivMilpReturn = collections.namedtuple("LyapDerivMilpReturn", [
            "milp", "x", "beta", "gamma", "x_next", "s", "z", "z_next",
            "beta_next", "system_constraint_return",
            "lyap_relu_x_mip_cnstr_ret", "lyap_relu_x_next_mip_cnstr_ret"
        ])
        return LyapDerivMilpReturn(
            milp=milp,
            x=x,
            beta=beta,
            gamma=gamma,
            x_next=x_next,
            s=s,
            z=z,
            z_next=z_next,
            beta_next=beta_next,
            system_constraint_return=system_constraint_return,
            lyap_relu_x_mip_cnstr_ret=lyap_relu_x_mip_cnstr_ret,
            lyap_relu_x_next_mip_cnstr_ret=lyap_relu_x_next_mip_cnstr_ret)

    def strengthen_lyapunov_derivative_milp_binary(self,
                                                   lyap_deriv_milp_return,
                                                   gurobi_options=None):
        """
        Given an MILP that verifies the Lyapunov derivative condition, we want
        to strengthen this MILP formulation, by putting constraints on its
        binary variables. Specifically, we check if the binary variable
        (beta_next) for the Lyapunov network computing V(x_next) and the
        binary variable beta for the Lyapunov network computing V(x) should be
        both active or both inactive. We do so by adding the constraint
        beta(i) + beta_next(i) = 1. If the MILP is infeasible with this
        constraint, then it means beta(i) and beta_next(i) have to be both
        active or both inactive, hence we strengthen the MILP by adding the
        constraint beta(i) = beta_next(i)
        @param lyap_deriv_milp_return [in/out] The return from
        lyapunov_derivative_as_milp() function.
        """
        assert (len(lyap_deriv_milp_return.beta) == len(
            lyap_deriv_milp_return.beta_next))
        # We know that the derivative MILP always has a solution of x=x*, such
        # that the objective is 0. Hence we put the constraint that the
        # objective should be non-negative, namely we focus on only the
        # adversarial states.
        objective = lyap_deriv_milp_return.milp.gurobi_model.getObjective()
        nonnegative_objective_cnstr = \
            lyap_deriv_milp_return.milp.gurobi_model.addLConstr(objective >= 0)
        gurobi_default_options = dict()

        def set_gurobi_options(param, param_val):
            default_val = lyap_deriv_milp_return.milp.gurobi_model.\
                getParamInfo(param)[2]
            gurobi_default_options[param] = default_val
            lyap_deriv_milp_return.milp.gurobi_model.setParam(param, param_val)

        set_gurobi_options(gurobipy.GRB.Param.OutputFlag, False)
        set_gurobi_options(gurobipy.GRB.Param.DualReductions, False)
        # Terminate once it finds a single solution. We don't care
        # finding the optimal solution in this pre-solve stage.
        set_gurobi_options(gurobipy.GRB.Param.SolutionLimit, 1)
        if gurobi_options is not None:
            for param, param_val in gurobi_options.items():
                set_gurobi_options(param, param_val)
        for i in range(len(lyap_deriv_milp_return.beta)):
            # beta[i] and beta_next[i] can both take value 0 and 1.
            if lyap_deriv_milp_return.lyap_relu_x_mip_cnstr_ret.\
                relu_input_lo[i] < 0 and\
                lyap_deriv_milp_return.lyap_relu_x_mip_cnstr_ret.\
                relu_input_up[i] > 0 and\
                lyap_deriv_milp_return.lyap_relu_x_next_mip_cnstr_ret.\
                relu_input_lo[i] < 0 and\
                lyap_deriv_milp_return.lyap_relu_x_next_mip_cnstr_ret.\
                    relu_input_up[i] > 0:
                gurobi_cnstr = lyap_deriv_milp_return.milp.gurobi_model.\
                    addConstr(lyap_deriv_milp_return.beta[i] +
                              lyap_deriv_milp_return.beta_next[i] == 1)
                lyap_deriv_milp_return.milp.gurobi_model.optimize()
                if lyap_deriv_milp_return.milp.gurobi_model.status ==\
                        gurobipy.GRB.Status.INFEASIBLE:
                    # beta[i] and beta_next[i] must be both active or both
                    # inactive. Add constraint beta[i] = beta_next[i]
                    print(f"add constraint beta[{i}] = beta_next[{i}]")
                    lyap_deriv_milp_return.milp.addLConstr(
                        [
                            torch.tensor(
                                [1, -1],
                                dtype=lyap_deriv_milp_return.milp.dtype)
                        ], [[
                            lyap_deriv_milp_return.beta[i],
                            lyap_deriv_milp_return.beta_next[i]
                        ]],
                        sense=gurobipy.GRB.EQUAL,
                        rhs=0.)
                lyap_deriv_milp_return.milp.gurobi_model.remove(gurobi_cnstr)
        # Reset the gurobi params to default value.
        for param, param_default_val in gurobi_default_options.items():
            lyap_deriv_milp_return.milp.gurobi_model.setParam(
                param, param_default_val)
        # Remove the constraint that the objective should be non-negative.
        lyap_deriv_milp_return.milp.gurobi_model.remove(
            nonnegative_objective_cnstr)

    def strengthen_lyapunov_derivative_as_milp(self,
                                               x_equilibrium,
                                               V_lambda,
                                               epsilon,
                                               epsilon_type: ConvergenceEps,
                                               num_strengthen_pts,
                                               *,
                                               R,
                                               lyapunov_lower=None,
                                               lyapunov_upper=None,
                                               x_warmstart=None):
        """
        Strengthen the MILP for verifying Lyapunov derivative condition.

        The MILP from lyapunov_derivative_as_milp uses the big-M formulation.
        We can strengthen this MILP formulation with the following algorithm:
        1. First construct the LP relaxation of the MILP.
        2. Repeat for num_strengthen_pts:
            3. Solve this LP to optimality.
            4. For each neural network, strengthen its LP relaxation with the
               most violated ideal constraint evaluated at the LP solution.
        5. In the LP relaxation, changed the relaxed continuous variables back
           to binary variables.
        """
        # Step 1, create the LP relaxation.
        lyap_deriv_lp_return = self.lyapunov_derivative_as_milp(
            x_equilibrium,
            V_lambda,
            epsilon,
            epsilon_type,
            R=R,
            lyapunov_lower=lyapunov_lower,
            lyapunov_upper=lyapunov_upper,
            x_warmstart=x_warmstart,
            binary_var_type=gurobi_torch_mip.BINARYRELAX)
        for _ in range(num_strengthen_pts):
            # Step 3, solve the LP relaxation.
            lyap_deriv_lp_return.milp.gurobi_model.setParam(
                gurobipy.GRB.Param.OutputFlag, False)
            lyap_deriv_lp_return.milp.gurobi_model.optimize()
            assert (lyap_deriv_lp_return.milp.gurobi_model.status ==
                    gurobipy.GRB.Status.OPTIMAL)
            # Step 4, strengthen each neural network:
            # Strengthen the ReLU network in lyapunov function.
            self.lyapunov_relu_free_pattern.strengthen_relu_mip_at_solution(
                lyap_deriv_lp_return.milp, lyap_deriv_lp_return.x,
                lyap_deriv_lp_return.z, lyap_deriv_lp_return.beta,
                lyap_deriv_lp_return.lyap_relu_x_mip_cnstr_ret)
            self.lyapunov_relu_free_pattern.strengthen_relu_mip_at_solution(
                lyap_deriv_lp_return.milp, lyap_deriv_lp_return.x_next,
                lyap_deriv_lp_return.z_next, lyap_deriv_lp_return.beta_next,
                lyap_deriv_lp_return.lyap_relu_x_next_mip_cnstr_ret)
            # Strengthen the ReLU network in dynamics constraint.
            if (isinstance(self.system, feedback_system.FeedbackSystem)):
                self.system.strengthen_dynamics_constraint(
                    lyap_deriv_lp_return.milp, lyap_deriv_lp_return.
                    system_constraint_return.forward_dynamics_return,
                    lyap_deriv_lp_return.system_constraint_return.
                    controller_mip_cnstr_return)

        # Step 5 remove binary relaxation.
        lyap_deriv_lp_return.milp.remove_binary_relaxation()
        return lyap_deriv_lp_return

    def lyapunov_derivative_loss_at_samples(self,
                                            V_lambda,
                                            epsilon,
                                            state_samples,
                                            x_equilibrium,
                                            eps_type,
                                            *,
                                            R,
                                            margin=0.,
                                            reduction="mean",
                                            weight=None):
        """
        We will sample states xⁱ, i=1,...N, and we would like the Lyapunov
        function to decrease on these sampled states xⁱ. We denote l(x) as the
        function we want to penalize, and define a loss as
        mean(max(l(xⁱ) + margin, 0))
        Depending on eps_type, l is defined as
        1. If we want to prove the exponential convergence rate is larger than
           epsilon, then l(x) = V(x_next) - V(x) + ε*V(x)
        2. If we want to prove the exponential convergence rate is smaller
           than epsilon, then l(x) = -(V(x_next) - V(x) + ε*V(x))
        3. If we want to prove the asymptotic convergence, then
           l(x) = V(x_next) - V(x) + ε*|R*(x−x*)|₁
        @param V_lambda λ in the Lyapunov function.
        @param epsilon ε in the Lyapunov function.
        @param state_samples The sampled state x[n], state_samples[i] is the
        i'th sample xⁱ[n]
        @param x_equilibrium x*.
        @param eps_type The interpretation of epsilon. Whether we prove
        exponential or asymptotic convergence.
        @param margin We might want to shift the margin for the Lyapunov
        loss.
        @param reduction If reduction=mean, then use the mean loss, otherwise
        use the max over all samples.
        @param weight If set to None, then we use uniform weight of 1 for
        every sample. Otherwise weight should be a vector of the same length
        as the number of samples, whereh weight[i] is the weight of
        state_samples[i].
        @return loss The loss
        mean(max(V(xⁱ[n+1]) - V(xⁱ[n]) + ε*V(xⁱ[n]) + margin, 0))
        """
        assert (isinstance(V_lambda, float))
        assert (isinstance(epsilon, float))
        assert (isinstance(state_samples, torch.Tensor))
        assert (state_samples.shape[1] == self.system.x_dim)
        assert (isinstance(eps_type, ConvergenceEps))
        R = _get_R(R, self.system.x_dim, state_samples.device)
        state_next = self.system.step_forward(state_samples)

        return self.lyapunov_derivative_loss_at_samples_and_next_states(
            V_lambda,
            epsilon,
            state_samples,
            state_next,
            x_equilibrium,
            eps_type,
            R=R,
            margin=margin,
            reduction=reduction,
            weight=weight)

    def lyapunov_derivative_loss_at_samples_and_next_states(
            self,
            V_lambda,
            epsilon,
            state_samples,
            state_next,
            x_equilibrium,
            eps_type,
            *,
            R,
            margin=0.,
            reduction="mean",
            weight=None):
        """
        We will sample states xⁱ, i=1,...N, and we would like the Lyapunov
        function to decrease on these sampled states xⁱ. We denote l(x) as the
        function we want to penalize, and define a loss as
        mean(max(l(xⁱ) + margin, 0))
        Depending on eps_type, l is defined as
        1. If we want to prove the exponential convergence rate is larger than
           epsilon, then l(x) = V(x_next) - V(x) + ε*V(x)
        2. If we want to prove the exponential convergence rate is smaller
           than epsilon, then l(x) = -(V(x_next) - V(x) + ε*V(x))
        3. If we want to prove the asymptotic convergence, then
           l(x) = V(x_next) - V(x) + ε*|R*(x−x*)|₁
        The lyapunov function is
        ReLU(x) - ReLU(x*) + λ|R*(x-x*)|₁
        @param V_lambda λ in the Lyapunov function.
        @param epsilon ε in the Lyapunov function.
        @param state_samples The sampled state x[n], state_samples[i] is the
        i'th sample xⁱ[n]
        @param state_next The next state x[n+1], state_next[i] is the next
        state for the i'th sample xⁱ[n+1]
        @param x_equilibrium x*.
        @param exp_type The interpretation of epsilon. If exp_type=ExpLower,
        then the loss wrt to the convergence lower bound. If exp_type=ExpUpper,
        then the loss is with respect to the upper bound
        @param margin We might want to shift the margin for the Lyapunov
        loss.
        @param weight If set to None, then we use uniform weight of 1 for
        every sample. Otherwise weight should be a vector of the same length
        as the number of samples, whereh weight[i] is the weight of
        state_samples[i].
        @return loss The loss
        mean(max(V(xⁱ[n+1]) - V(xⁱ[n]) + ε*V(xⁱ[n]) + margin, 0))
        """
        assert (isinstance(V_lambda, float))
        assert (isinstance(epsilon, float))
        assert (isinstance(state_samples, torch.Tensor))
        assert (state_samples.shape[1] == self.system.x_dim)
        assert (isinstance(state_next, torch.Tensor))
        assert (state_next.shape[1] == self.system.x_dim)
        assert (state_samples.shape[0] == state_next.shape[0])
        assert (isinstance(eps_type, ConvergenceEps))
        assert (reduction in {"mean", "max", "4norm"})
        R = _get_R(R, self.system.x_dim, state_samples.device)
        v1 = self.lyapunov_value(state_samples, x_equilibrium, V_lambda, R=R)
        v2 = self.lyapunov_value(state_next, x_equilibrium, V_lambda, R=R)

        if eps_type == ConvergenceEps.ExpLower:
            hinge_loss_all = torch.nn.HingeEmbeddingLoss(
                margin=margin,
                reduction="none")(-(v2 - v1 + epsilon * v1),
                                  torch.tensor(-1.).to(state_samples.device))
        elif eps_type == ConvergenceEps.ExpUpper:
            hinge_loss_all = torch.nn.HingeEmbeddingLoss(
                margin=margin,
                reduction="none")((v2 - v1 + epsilon * v1),
                                  torch.tensor(-1.).to(state_samples.device))
        elif eps_type == ConvergenceEps.Asymp:
            hinge_loss_all = torch.nn.HingeEmbeddingLoss(
                margin=margin,
                reduction="none")(-(v2 - v1 + epsilon * torch.norm(
                    R @ (state_samples - x_equilibrium).T, p=1, dim=0)),
                                  torch.tensor(-1.).to(state_samples.device))
        else:
            raise Exception("Unknown eps_type")
        if reduction == "mean":
            if weight is None:
                return torch.mean(hinge_loss_all)
            else:
                assert (weight.shape == (state_samples.shape[0], ))
                return torch.mean(weight * hinge_loss_all)
        elif reduction == "max":
            return torch.max(hinge_loss_all)
        elif reduction == "4norm":
            return torch.norm(hinge_loss_all, p=4)

    def compute_region_of_attraction(self, V_lambda, R, x_equilibrium,
                                     V_upper_bound, x_lo_larger, x_up_larger):
        """
        After we have found the Lyapunov function satisfying the positivity and
        derivative conditions, i.e., V(x) > 0 and dV(x) < 0 for all
        x_lo <= x <= x_up and V(x) < V_upper_bound, we then want to find the
        region-of-attraction certified by this Lyapunov function. The region
        of attraction is the biggest sub-level set contained in the region
        S = {x | x_lo <= x <= x_up and V(x) < V_upper_bound}. Namely we can
        solve the following problems to find the biggest sub-level set
        obj1 = min V(x[n])
        s.t x[n] ∈ B, x[n+1] ∉ B
        obj2 = min V(x[n])
        s.t x[n] ∉ B, x[n+1] ∈ B
        where B = {x | x_lo <= x <= x_up}.

        And the region-of-attraction is
        {x | V(x) < min(obj1, obj2, V_upper_bound)}.

        We will write the constraint x∉B as
        ∃i, s.t x_up(i) < x(i) < x_up_larger(i) or
                x_lo_larger(i) < x(i) < x_lo(i)

        Note that currently we only support the verification region being an
        axis-aligned bounding box x_lo <= x <= x_up. The more complicated
        verification region is not supported yet.
        Note that our Lyapunov function is V(x) = ϕ(x) − ϕ(x*) + λ |R(x−x*)|₁
        @param V_lambda λ in our Lyapunov function.
        @param R R in the Lyapunov function
        @param x_equilibrium x* in the Lyapunov function.
        @param V_upper_bound We verified the Lyapunov condition with
        V(x) < V_upper_bound. Set to None of inf to ignore V_upper_bound.
        @param x_lo_larger We use x_lo_larger to write a bounded region larger
        than x_lo <= x <= x_up
        @param x_up_larger We use x_lo_larger to write a bounded region larger
        than x_lo <= x <= x_up
        """
        milp1, _, _, _, _ = self._construct_milp_for_roa(
            V_lambda, R, x_equilibrium, x_lo_larger, x_up_larger, True)
        milp1.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp1.gurobi_model.optimize()
        obj1 = milp1.gurobi_model.ObjVal if milp1.gurobi_model.status ==\
            gurobipy.GRB.Status.OPTIMAL else np.inf

        milp2, _, _, _, _ = self._construct_milp_for_roa(
            V_lambda, R, x_equilibrium, x_lo_larger, x_up_larger, False)
        milp2.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp2.gurobi_model.optimize()
        obj2 = milp2.gurobi_model.ObjVal if milp2.gurobi_model.status ==\
            gurobipy.GRB.Status.OPTIMAL else np.inf

        if V_upper_bound is None:
            V_upper_bound = np.inf
        return np.min([obj1, obj2, V_upper_bound])

    def _construct_milp_for_roa(self, V_lambda, R, x_equilibrium, x_lo_larger,
                                x_up_larger, x_curr_in_box: bool):
        """
        This is the internal function to formulate an MILP for computing the
        region of attraction (ROA). Refer to compute_region_of_attraction for
        more information.
        """
        dtype = self.system.dtype
        # I could use the original gurobi interface directly, instead of the
        # gurobi_torch_mip interface, as we don't need to compute the gradient
        # of the results.
        milp = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x_curr = milp.addVars(self.system.x_dim,
                              lb=-gurobipy.GRB.INFINITY,
                              vtype=gurobipy.GRB.CONTINUOUS,
                              name="x[n]")
        x_next = milp.addVars(self.system.x_dim,
                              lb=-gurobipy.GRB.INFINITY,
                              vtype=gurobipy.GRB.CONTINUOUS,
                              name="x[n+1]")

        # Add the dynamics constraint between x[n+1] and x[n]
        dynamic_system._add_system_constraint(self.system, milp, x_curr,
                                              x_next)

        if x_curr_in_box:
            out_box_x = x_next
        else:
            out_box_x = x_curr

        if not x_curr_in_box:
            # Add the constraint x_lo <= x[n+1] <= x_up
            milp.addMConstr([torch.eye(self.system.x_dim, dtype=dtype)],
                            [x_next],
                            sense=gurobipy.GRB.LESS_EQUAL,
                            b=torch.from_numpy(self.system.x_up_all))
            milp.addMConstr([-torch.eye(self.system.x_dim, dtype=dtype)],
                            [x_next],
                            sense=gurobipy.GRB.LESS_EQUAL,
                            b=torch.from_numpy(-self.system.x_lo_all))
        z, beta, a_out, b_out, _ = self.add_lyap_relu_output_constraint(
            milp, x_curr)

        # Now add the constraint that x is outside of the box
        # x_lo <= x <= x_up but within the region
        # x_lo_larger <= x <= x_up_larger.
        # This region could be written as the union of boxes
        # box i: x_up(i) <= x(i) <= x_up_larger(i)
        #        x_lo(j) <= x(j) <= x_up(j) ∀ j≠ i
        # box x_dim + i: x_lo_larger(i) <= x(i) <= x_lo(i)
        #                       x_lo(j) <= x(j) <= x_up(j) ∀ j≠ i
        # If we denote the bounds for the i'th box as
        # x_box_lo_i <= x <= x_box_up_i
        # We need to introduce 2 * x_dim binary variables ζ to determine in
        # which box x lives, with the constraints
        # x = t₁ + ... tₘ
        # x_box_lo_i * ζᵢ <= tᵢ <= x_box_up_i * ζᵢ
        # ζ₁ + ... + ζₘ = 1
        # where m = 2 * x_dim
        box_zeta = milp.addVars(2 * self.system.x_dim,
                                lb=-gurobipy.GRB.INFINITY,
                                vtype=gurobipy.GRB.BINARY,
                                name="box_zeta")
        # Add constraint ζ₁ + ... + ζₘ = 1
        milp.addLConstr([torch.ones((2 * self.system.x_dim, ), dtype=dtype)],
                        [box_zeta],
                        sense=gurobipy.GRB.EQUAL,
                        rhs=1.)
        t_slack = [None] * 2 * self.system.x_dim
        for i in range(2 * self.system.x_dim):
            t_slack[i] = milp.addVars(self.system.x_dim,
                                      lb=-gurobipy.GRB.INFINITY,
                                      vtype=gurobipy.GRB.CONTINUOUS,
                                      name=f"t_slack[{i}]")
            x_box_lo_i = torch.from_numpy(self.system.x_lo_all).clone()
            x_box_up_i = torch.from_numpy(self.system.x_up_all).clone()
            if i < self.system.x_dim:
                x_box_lo_i[i] = self.system.x_up_all[i]
                x_box_up_i[i] = x_up_larger[i]
            else:
                x_box_lo_i[i - self.system.x_dim] = \
                    x_lo_larger[i-self.system.x_dim]
                x_box_up_i[i-self.system.x_dim] = \
                    self.system.x_lo_all[i-self.system.x_dim]
            # Now add the constraint
            # x_box_lo_i * ζᵢ <= tᵢ <= x_box_up_i * ζᵢ
            milp.addMConstr([
                x_box_lo_i.reshape(
                    (-1, 1)), -torch.eye(self.system.x_dim, dtype=dtype)
            ], [[box_zeta[i]], t_slack[i]],
                            sense=gurobipy.GRB.LESS_EQUAL,
                            b=torch.zeros((self.system.x_dim, ), dtype=dtype))
            milp.addMConstr([
                torch.eye(self.system.x_dim, dtype=dtype), -x_box_up_i.reshape(
                    (-1, 1))
            ], [t_slack[i], [box_zeta[i]]],
                            sense=gurobipy.GRB.LESS_EQUAL,
                            b=torch.zeros((self.system.x_dim, ), dtype=dtype))
        # Add constraint x = t₁ + ... tₘ
        milp.addMConstr([torch.eye(self.system.x_dim, dtype=dtype)] +
                        [-torch.eye(self.system.x_dim, dtype=dtype)] *
                        (2 * self.system.x_dim), [out_box_x] + t_slack,
                        sense=gurobipy.GRB.EQUAL,
                        b=torch.zeros((self.system.x_dim, ), dtype=dtype))

        relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)

        # Now write the 1-norm |R*(x[n] - x*)|₁ as mixed-integer linear
        # constraints.
        (s,
         gamma) = self.add_state_error_l1_constraint(milp,
                                                     x_equilibrium,
                                                     x_curr,
                                                     R=R,
                                                     slack_name="s",
                                                     binary_var_name="gamma")

        # Objective is
        # min V(x[n])
        # = ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁
        # = a_out * z + b_out + λ * s - ϕ(x*)
        milp.setObjective(
            [a_out.squeeze(), V_lambda * torch.ones(
                (len(s), ), dtype=dtype)], [z, s],
            constant=b_out - relu_at_equilibrium.squeeze(),
            sense=gurobipy.GRB.MINIMIZE)
        return milp, x_curr, x_next, t_slack, box_zeta


def _get_R(R, x_dim: int, device):
    """
    Take matrix R used in the 1-norm |R*(x-x*)|₁.
    """
    assert (isinstance(R, torch.Tensor) or R is None)
    if R is None:
        return torch.eye(x_dim, dtype=torch.float64).to(device)
    else:
        assert (R.shape[1] == x_dim)
        return R.to(device)
