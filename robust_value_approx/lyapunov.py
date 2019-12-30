# -*- coding: utf-8 -*-
import gurobipy
import torch

import robust_value_approx.relu_to_optimization as relu_to_optimization
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip
import robust_value_approx.utils as utils


class LyapunovDiscreteTimeHybridSystem:
    """
    For a discrete time autonomous hybrid linear system
    x[n+1] = Aᵢ*x[n] + cᵢ
    if Pᵢ * x[n] <= qᵢ
    i = 1, ..., K.
    we want to learn a ReLU network as the Lyapunov function for the system.
    The condition for the Lyapunov function is that
    V[x[n+1]] <= V[x[n]] ∀x[n]
    We will first formulate this condition as the optimal cost of a certain
    mixed-integer linear program (MILP) being non-positive. The optimal cost
    is the loss function of our neural network. We will compute the gradient of
    this loss function w.r.t to network weights/bias, and then call gradient
    based optimization (SGD/Adam) to reduce the loss.
    """

    def __init__(self, system):
        """
        @param system A AutonomousHybridLinearSystem instance.
        """
        assert(isinstance(
            system, hybrid_linear_system.AutonomousHybridLinearSystem))
        self.system = system

    def lyapunov_value(
            self, relu_model, x, x_equilibrium, V_rho,
            relu_at_equilibrium=None):
        """
        Compute the value of the Lyapunov function as
        ReLU(x) - ReLU(x*) + ρ|x-x*|₁
        where |x-x*|₁ is the 1-norm of x-x*.
        @param relu_model A ReLU (including leaky relu) model.
        @param x a torch tensor. Evaluate Lyapunov function at this point.
        @param x_equilibrium a torch tensor. The equilibrium state x*.
        @param V_rho ρ in the documentation above.
        @param relu_at_equilibrium. ReLU(x*). If set to None, then we compute
        ReLU(x*) in this function.
        """
        if relu_at_equilibrium is None:
            relu_at_equilibrium = relu_model.forward(x_equilibrium)
        return relu_model.forward(x) - relu_at_equilibrium +\
            V_rho * torch.norm(x - x_equilibrium, p=1)

    def lyapunov_positivity_as_milp(
            self, relu_model, x_equilibrium, V_rho, V_epsilon):
        """
        For a ReLU network, in order to determine if the function
        V(x) = ReLU(x) - ReLU(x*) + ρ * |x - x*|₁
        where |x - x*|₁ is the 1-norm of the vector x - x*.
        satisfies the positivity constraint of Lyapunov condition
        V(x) > 0 ∀ x ≠ x*
        We check a strong condition
        V(x) ≥ ε |x - x*|₁ ∀ x
        where ε is a small positive number.To check if the stronger condition
        is satisfied, we can solve the following optimization problem
        min x ReLU(x) - ReLU(x*) + (ρ-ε) * |x - x*|₁
        We can formulate this optimization problem as a mixed integer linear
        program, solve the for optimal solution of this program. If the optimal
        cost is no smaller than 0, then we proved the positivity constraint
        V(x) > 0 ∀ x ≠ x*
        @param relu_model A ReLU pytorch model.
        @param x_equilibrium The equilibrium state x*.
        @param V_rho ρ in the documentation above.
        @param V_epsilon A scalar. ε in the documentation above.
        @return (milp, x) milp is a GurobiTorchMILP instance, x is the decision
        variable for state.
        """
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (self.system.x_dim,))
        assert(isinstance(V_rho, float))
        assert(isinstance(V_epsilon, float))
        relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            relu_model, self.system.dtype)

        dtype = self.system.dtype
        milp = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x")
        # z is the slack variable to write the output of ReLU network as mixed
        # integer constraints.
        Ain_x, Ain_z, Ain_beta, rhs_in, Aeq_x, Aeq_z, Aeq_beta, rhs_eq, a_out,\
            b_out, _, _, _, _ = relu_free_pattern.output_constraint(
                relu_model, torch.from_numpy(self.system.x_lo_all),
                torch.from_numpy(self.system.x_up_all))
        z = milp.addVars(
            Ain_z.shape[1], lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="z")
        beta = milp.addVars(
            Ain_beta.shape[1], vtype=gurobipy.GRB.BINARY, name="beta")
        for i in range(Ain_x.shape[0]):
            milp.addLConstr(
                [Ain_x[i], Ain_z[i], Ain_beta[i]], [x, z, beta],
                sense=gurobipy.GRB.LESS_EQUAL, rhs=rhs_in[i])
        for i in range(Aeq_x.shape[0]):
            milp.addLConstr([Aeq_x[i], Aeq_z[i], Aeq_beta[i]], [x, z, beta],
                            sense=gurobipy.GRB.EQUAL, rhs=rhs_eq[i])

        # Now write the 1-norm |x - x*|₁ as mixed-integer linear constraints.
        # TODO(hongkai.di): support the case when x_equilibrium is not in the
        # strict interior of the state space.
        if not torch.all(torch.from_numpy(self.system.x_lo_all) <
                         x_equilibrium) or\
                not torch.all(torch.from_numpy(self.system.x_up_all) >
                              x_equilibrium):
            raise Exception("lyapunov_positivity_as_milp: we currently " +
                            "require that x_lo < x_equilibrium < x_up")
        s = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="s")
        gamma = milp.addVars(
            self.system.x_dim, vtype=gurobipy.GRB.BINARY, name="gamma")

        for i in range(self.system.x_dim):
            if self.system.x_lo_all[i] < x_equilibrium[i] and\
                    self.system.x_up_all[i] > x_equilibrium[i]:
                Ain_x2, Ain_s2, Ain_gamma2, rhs_in2 =\
                    utils.replace_absolute_value_with_mixed_integer_constraint(
                        self.system.x_lo_all[i] - x_equilibrium[i],
                        self.system.x_up_all[i] - x_equilibrium[i],
                        dtype=dtype)
                for j in range(Ain_x2.shape[0]):
                    milp.addLConstr(
                        [torch.tensor([Ain_x2[j], Ain_s2[j], Ain_gamma2[j]],
                                      dtype=self.system.dtype)],
                        [[x[i], s[i], gamma[i]]],
                        sense=gurobipy.GRB.LESS_EQUAL,
                        rhs=rhs_in2[j] + Ain_x2[j] * x_equilibrium[i])
        # Now compute ReLU(x*)
        relu_x_equilibrium = relu_model.forward(x_equilibrium)

        milp.setObjective(
            [a_out,
             (V_rho-V_epsilon)*torch.ones((self.system.x_dim,), dtype=dtype)],
            [z, s], constant=b_out - relu_x_equilibrium.squeeze(),
            sense=gurobipy.GRB.MINIMIZE)
        return (milp, x)

    def lyapunov_positivity_loss_at_sample(
            self, relu_model, relu_at_equilibrium, x_equilibrium, state_sample,
            V_rho, margin=0.):
        """
        We will sample a state xⁱ, and we would like the Lyapunov function to
        be larger than 0 at xⁱ. Hence we define the loss as
        max(-V(xⁱ) + margin, 0)
        @param relu_model  The Lyapunov function is
        ReLU(x) - ReLU(x*) + ρ|x-x*|₁
        @param relu_at_equilibrium A 0-D tensor. ReLU(x*)
        @param x_equilibrium x* in the documentation above.
        @param state_sample The sampled state xⁱ.
        @param V_rho ρ in the documentation above.
        @param margin The margin used in the hinge loss.
        """
        assert(isinstance(relu_at_equilibrium, torch.Tensor))
        assert(isinstance(state_sample, torch.Tensor))
        assert(state_sample.shape == (self.system.x_dim,))
        assert(isinstance(state_sample, torch.Tensor))
        assert(x_equilibrium.shape == (self.system.x_dim,))
        assert(isinstance(V_rho, float))
        assert(isinstance(margin, float))
        return torch.nn.HingeEmbeddingLoss(margin=margin)(
            self.lyapunov_value(
                relu_model, state_sample, x_equilibrium, V_rho,
                relu_at_equilibrium), torch.tensor(-1.))

    def lyapunov_derivative_as_milp(
            self, relu_model, x_equilibrium, V_rho, epsilon,
            lyapunov_lower=None, lyapunov_upper=None):
        """
        We assume that the Lyapunov function
        V(x) = ReLU(x) - ReLU(x*) + ρ|x-x*|₁, where x* is the equilibrium
        state.
        Formulate the Lyapunov condition
        V(x[n+1]) - V(x[n]) <= -ε * V(x[n]) ∀x[n] satisfying
        lower <= V(x[n]) <= upper
        as the maximal of following optimization problem is no larger
        than 0.
        max V(x[n+1]) - V(x[n]) + ε * V(x[n])
        s.t lower <= V(x[n]) <= upper
        Notice that to prove global stability, then lower = 0 and upper = inf,
        the optimal has to be strictly less than 0 except at the equilibrium.
        If lower > 0 and the maximal is strictly less than 0, then we prove
        that all states outside of the level set {x[n] | V(x[n]) <= lower} will
        converge to this level set.
        We would formulate this optimization problem as an MILP.

        This function returns the MILP formulation.
        max cᵣᵀ * r + c_zetaᵀ * ζ + c_constant
        s.t Ain_r * r + Ain_zeta * ζ <= rhs_in
            Aeq_r * r + Aeq_zeta * ζ = rhs_eq
        where r includes all continuous variables, and ζ includes all binary
        variables.
        @param relu_model A pytorch ReLU network. Notice that we want the last
        layer to be a ReLU activation layer, so as to guarantee the Lyapunov
        function to be non-negative.
        return (milp, x, x_next, s, gamma, z, z_next, beta, beta_next)
        where milp is a GurobiTorchMILP object.
        The decision variables of the MILP are
        (x[n], x[n+1], s[n], gamma[n], z[n], z[n+1], beta[n], beta[n+1])
        @param relu_model The Lyapunov function is represented as the output of
        a ReLU model.
        @param V_rho ρ in the documentation above.
        @param lyapunov_lower the "lower" bound in the documentation above. If
        lyapunov_lower = None, then we ignore the lower bound on V(x[n]).
        @param lyapunov_upper the "upper" bound in the documentation above. If
        lyapunov_upper = None, then we ignore the upper bound on V(x[n]).
        @param epsilon The rate of exponential convergence. If the goal is to
        verify convergence but not exponential convergence, then set epsilon
        to 0.
        """
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (self.system.x_dim,))
        if lyapunov_lower is not None:
            assert(isinstance(lyapunov_lower, float))
        if lyapunov_upper is not None:
            assert(isinstance(lyapunov_upper, float))
        assert(isinstance(V_rho, float))
        assert(isinstance(epsilon, float))

        relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            relu_model, self.system.dtype)

        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)

        # x is the variable x[n]
        x = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x[n]")
        # x_next is the variable x[n+1]
        x_next = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x[n+1]")
        # s is the slack variable to convert hybrid linear system to
        # mixed-integer linear constraint.
        s = milp.addVars(
            self.system.x_dim * self.system.num_modes,
            lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS, name="s")
        # gamma is the binary variable determining the hybrid mode of x[n]
        gamma = milp.addVars(
            self.system.num_modes, lb=0, vtype=gurobipy.GRB.BINARY,
            name="gamma")
        # Now add the milp constraint to formulate the hybrid linear system.
        (Aeq_s1, Aeq_gamma1, Ain_x1, Ain_s1, Ain_gamma1, rhs_in1) =\
            self.system.mixed_integer_constraints()
        # Add the constraint x[n+1] = Aeq_s1 * s + Aeq_gamma1 * gamma
        for i in range(self.system.x_dim):
            milp.addLConstr(
                [torch.tensor([1.], dtype=milp.dtype), -Aeq_s1[i],
                 -Aeq_gamma1[i]], [[x_next[i]], s, gamma],
                sense=gurobipy.GRB.EQUAL, rhs=0.)
        # Now add the constraint
        # Ain_x1 * x + Ain_s1 * s + Ain_gamma1 * gamma <= rhs_in1
        for i in range(Ain_x1.shape[0]):
            milp.addLConstr(
                [Ain_x1[i], Ain_s1[i], Ain_gamma1[i]], [x, s, gamma],
                sense=gurobipy.GRB.LESS_EQUAL, rhs=rhs_in1[i])
        # Now add the constraint that sum gamma = 1
        milp.addLConstr(
            [torch.ones((self.system.num_modes,))], [gamma],
            sense=gurobipy.GRB.EQUAL, rhs=1.)

        # Add the mixed-integer constraint that formulates the output of
        # ReLU(x[n]).
        # z is the slack variable to write the output of ReLU(x[n]) with mixed
        # integer linear constraints.
        (Ain_x2, Ain_z, Ain_beta, rhs_in2, Aeq_x2, Aeq_z, Aeq_beta, rhs_eq2,
         a_out, b_out, z_lo, z_up, _, _) = relu_free_pattern.output_constraint(
             relu_model, torch.from_numpy(self.system.x_lo_all),
             torch.from_numpy(self.system.x_up_all))
        z = milp.addVars(
            Ain_z.shape[1], lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="z[n]")
        beta = milp.addVars(Ain_beta.shape[1],
                            vtype=gurobipy.GRB.BINARY, name="beta[n]")

        for i in range(Ain_x2.shape[0]):
            milp.addLConstr(
                [Ain_x2[i], Ain_z[i], Ain_beta[i]], [x, z, beta],
                sense=gurobipy.GRB.LESS_EQUAL, rhs=rhs_in2[i],
                name="milp_relu_xn[" + str(i) + "]")

        for i in range(Aeq_x2.shape[0]):
            milp.addLConstr(
                [Aeq_x2[i], Aeq_z[i], Aeq_beta[i]], [x, z, beta],
                sense=gurobipy.GRB.EQUAL, rhs=rhs_eq2[i],
                name="milp_relu_xn[" + str(i) + "]")

        # Now compute ReLU(x*)
        relu_x_equilibrium = relu_model.forward(x_equilibrium)

        # Now add the mixed-integer linear constraint to represent
        # |x[n] - x*|₁. To do so, we introduce the slack variable
        # s_x_norm, beta_x_norm.
        # s_x_norm(i) = |x[n](i) - x*(i)|
        s_x_norm = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="|x[n]-x*|")
        beta_x_norm = milp.addVars(
            self.system.x_dim, vtype=gurobipy.GRB.BINARY, name="beta_x_norm")
        # Now add the mixed-integer linear constraint to represent
        # |x[n+1] - x*|₁. To do so, we introduce the slack variable
        # s_x_next_norm, beta_x_next_norm.
        # s_x_next_norm(i) = |x[n+1](i) - x*(i)|
        s_x_next_norm = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="|x[n+1]-x*|")
        beta_x_next_norm = milp.addVars(
            self.system.x_dim, vtype=gurobipy.GRB.BINARY,
            name="beta_x_next_norm")
        for i in range(self.system.x_dim):
            Ain_x_norm, Ain_s_x_norm, Ain_beta_x_norm, rhs_x_norm = utils.\
                replace_absolute_value_with_mixed_integer_constraint(
                    self.system.x_lo_all[i] - x_equilibrium[i],
                    self.system.x_up_all[i] - x_equilibrium[i],
                    self.system.dtype)
            for j in range(Ain_x_norm.shape[0]):
                milp.addLConstr(
                    [torch.tensor([Ain_x_norm[j], Ain_s_x_norm[j],
                                   Ain_beta_x_norm[j]],
                                  dtype=self.system.dtype)],
                    [[x[i], s_x_norm[i], beta_x_norm[i]]],
                    rhs=rhs_x_norm[j] + Ain_x_norm[j]*x_equilibrium[i],
                    sense=gurobipy.GRB.LESS_EQUAL)
                milp.addLConstr(
                    [torch.tensor([Ain_x_norm[j], Ain_s_x_norm[j],
                                   Ain_beta_x_norm[j]],
                                  dtype=self.system.dtype)],
                    [[x_next[i], s_x_next_norm[i], beta_x_next_norm[i]]],
                    rhs=rhs_x_norm[j] + Ain_x_norm[j]*x_equilibrium[i],
                    sense=gurobipy.GRB.LESS_EQUAL)

        # Now add the constraint
        # lower <= ReLU(x[n]) - ReLU(x*) + ρ|x[n]-x*|₁ <= upper
        if lyapunov_lower is not None:
            milp.addLConstr(
                [a_out, V_rho * torch.ones((self.system.x_dim,),
                                           dtype=self.system.dtype)],
                [z, s_x_norm], sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=lyapunov_lower - b_out + relu_x_equilibrium)
        if lyapunov_upper is not None:
            milp.addLConstr(
                [a_out, V_rho * torch.ones((self.system.x_dim,),
                                           dtype=self.system.dtype)],
                [z, s_x_norm], sense=gurobipy.GRB.LESS_EQUAL,
                rhs=lyapunov_upper - b_out + relu_x_equilibrium)

        # Now write the ReLU output ReLU(x[n+1]) as mixed integer linear
        # constraints
        z_next = milp.addVars(Ain_z.shape[1], lb=-gurobipy.GRB.INFINITY,
                              vtype=gurobipy.GRB.CONTINUOUS, name="z[n+1]")
        beta_next = milp.addVars(Ain_beta.shape[1], lb=0.,
                                 vtype=gurobipy.GRB.BINARY, name="beta[n+1]")
        milp.gurobi_model.update()

        for i in range(Ain_x2.shape[0]):
            milp.addLConstr(
                [Ain_x2[i], Ain_z[i], Ain_beta[i]],
                [x_next, z_next, beta_next],
                sense=gurobipy.GRB.LESS_EQUAL, rhs=rhs_in2[i],
                name="milp_relu_xnext[" + str(i) + "]")
        for i in range(Aeq_x2.shape[0]):
            milp.addLConstr(
                [Aeq_x2[i], Aeq_z[i], Aeq_beta[i]],
                [x_next, z_next, beta_next],
                sense=gurobipy.GRB.EQUAL, rhs=rhs_eq2[i],
                name="milp_relu_xnext[" + str(i) + "]")

        # The cost function is
        # max ReLU(x[n+1]) + ρ|x[n+1]-x*|₁ - ReLU(x[n]) - ρ|x[n]-x*|₁ +
        #     epsilon * (ReLU(x[n]) - ReLU(x*) + ρ|x[n]-x*|₁)
        milp.setObjective(
            [a_out, (epsilon - 1) * a_out,
             V_rho * torch.ones((self.system.x_dim,), dtype=self.system.dtype),
             (epsilon-1) * V_rho*torch.ones((self.system.x_dim,),
                                            dtype=self.system.dtype)],
            [z_next, z, s_x_next_norm, s_x_norm],
            -epsilon * relu_x_equilibrium.squeeze(), gurobipy.GRB.MAXIMIZE)

        return (milp, x, x_next, s, gamma, z, z_next, beta, beta_next)

    def lyapunov_derivative_loss_at_sample(
            self, relu_model, V_rho, state_sample, x_equilibrium, margin=0.):
        """
        We will sample a state x̅[n], compute the next state x̅[n+1], and we
        would like the Lyapunov function to decrease on the sampled state
        x̅[n]. To do so, we define a loss as
        max(V(x̅[n+1]) - V(x̅[n]) + margin, 0)
        @param relu_model The lyapunov function is
        ReLU(x) - ReLU(x*) + ρ|x-x*|₁
        @param V_rho ρ in the Lyapunov function.
        @param state_sample The sampled state x̅[n]
        @param x_equilibrium x*.
        @param margin We might want to shift the margin for the Lyapunov
        loss. For example, Lyapunov condition requires V(x[n+1]) - V(x[n]) to
        be strictly negative for all x[n]. To do so, we can set margin to
        be a positive number
        @return loss The loss max(V(x̅[n+1]) - V(x̅[n]) + margin, 0)
        """
        assert(isinstance(V_rho, float))
        assert(isinstance(state_sample, torch.Tensor))
        assert(state_sample.shape == (self.system.x_dim,))
        # First compute the next state x̅[n+1]
        mode = self.system.mode(state_sample)
        if mode is None:
            raise Exception(
                "lyapunov_derivative_loss_at_sample: the input state_sample" +
                " is not in any mode of the hybrid system.")
        state_next = self.system.step_forward(state_sample, mode)

        return self.lyapunov_derivative_loss_at_sample_and_next_state(
            relu_model, V_rho, state_sample, state_next, x_equilibrium, margin)

    def lyapunov_derivative_loss_at_sample_and_next_state(
            self, relu_model, V_rho, state_sample, state_next, x_equilibrium,
            margin=0.):
        """
        We will sample a state x̅[n], compute the next state x̅[n+1], and we
        would like the Lyapunov function to decrease on the sampled state
        x̅[n]. To do so, we define a loss as
        max(V(x̅[n+1]) - V(x̅[n]) + margin, 0)
        @param relu_model The output of the ReLU model is the Lyapunov function
        value.
        @param V_rho ρ in Lyapunov function.
        @param state_sample The sampled state x̅[n]
        @param state_next The next state x̅[n+1]
        @param margin We might want to shift the margin for the Lyapunov
        loss. For example, Lyapunov condition requires V(x[n+1]) - V(x[n]) to
        be strictly negative for all x[n]. To do so, we can set margin to
        be a positive number
        @return loss The loss max(V(x̅[n+1]) - V(x̅[n]) + margin, 0)
        """
        assert(isinstance(V_rho, float))
        assert(isinstance(state_sample, torch.Tensor))
        assert(state_sample.shape == (self.system.x_dim,))
        assert(isinstance(state_next, torch.Tensor))
        assert(state_next.shape == (self.system.x_dim,))
        v1 = relu_model.forward(state_sample) +\
            V_rho * torch.norm(state_sample - x_equilibrium, p=1)
        v2 = relu_model.forward(state_next) + \
            V_rho * torch.norm(state_next - x_equilibrium, p=1)
        return torch.nn.HingeEmbeddingLoss(margin=margin)(
            v1 - v2, torch.tensor(-1.))
