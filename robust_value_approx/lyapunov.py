# -*- coding: utf-8 -*-
import gurobipy
import torch

import robust_value_approx.relu_to_optimization as relu_to_optimization
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.gurobi_torch_mip as gurobi_torch_mip
import robust_value_approx.utils as utils


class LyapunovHybridLinearSystem:
    """
    This is the super class of LyapunovDiscreteTimeHybridSystem and
    LyapunovContinuousTimeHybridSystem. It implements the common part of these
    two subclasses.
    """

    def __init__(self, system):
        """
        @param system A AutonomousHybridLinearSystem instance.
        """
        assert(isinstance(
            system, hybrid_linear_system.AutonomousHybridLinearSystem))
        self.system = system

    def add_hybrid_system_constraint(self, milp):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add the constraint and variables to write the hybrid linear system
        dynamics as mixed-integer linear constraints.
        """
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        x = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x")
        # s is the slack variable to convert hybrid linear system to
        # mixed-integer linear constraint. s[i*x_dim:(i+1)*x_dim] is the state
        # in the i'th mode.
        s = milp.addVars(
            self.system.x_dim * self.system.num_modes,
            lb=-gurobipy.GRB.INFINITY, vtype=gurobipy.GRB.CONTINUOUS, name="s")
        # gamma is the binary variable determining the hybrid mode of x.
        gamma = milp.addVars(
            self.system.num_modes, lb=0., vtype=gurobipy.GRB.BINARY,
            name="gamma")
        # Now add the milp constraint to formulate the hybrid linear system.
        (Aeq_s1, Aeq_gamma1, Ain_x1, Ain_s1, Ain_gamma1, rhs_in1) =\
            self.system.mixed_integer_constraints()

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
        return (x, s, gamma, Aeq_s1, Aeq_gamma1)

    def add_relu_output_constraint(
            self, relu_model, relu_free_pattern, milp, x):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add the relu output as mixed-integer linear constraint.
        """
        assert(isinstance(
            relu_free_pattern, relu_to_optimization.ReLUFreePattern))
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        assert(isinstance(x, list))
        (Ain_relu_x, Ain_relu_z, Ain_relu_beta, rhs_relu_in, Aeq_relu_x,
         Aeq_relu_z, Aeq_relu_beta, rhs_relu_eq, a_relu_out, b_relu_out, _,
         _, _, _) = \
            relu_free_pattern.output_constraint(
                 relu_model, torch.from_numpy(self.system.x_lo_all),
                 torch.from_numpy(self.system.x_up_all))
        # relu_z is the slack variable for the constraints encoding the relu
        # activation binary variable beta and the network input x.
        relu_z = milp.addVars(
            Ain_relu_z.shape[1], lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="relu_z")
        relu_beta = milp.addVars(
            Ain_relu_beta.shape[1], vtype=gurobipy.GRB.BINARY,
            name="relu_beta")
        for i in range(Ain_relu_x.shape[0]):
            milp.addLConstr(
                [Ain_relu_x[i], Ain_relu_z[i], Ain_relu_beta[i]],
                [x, relu_z, relu_beta], sense=gurobipy.GRB.LESS_EQUAL,
                rhs=rhs_relu_in[i], name="milp_relu")
        for i in range(Aeq_relu_x.shape[0]):
            milp.addLConstr(
                [Aeq_relu_x[i], Aeq_relu_z[i], Aeq_relu_beta[i]],
                [x, relu_z, relu_beta], sense=gurobipy.GRB.EQUAL,
                rhs=rhs_relu_eq[i], name="milp_relu")
        return (relu_z, relu_beta, a_relu_out, b_relu_out)

    def add_state_error_l1_constraint(
            self, milp, x_equilibrium, x, slack_name="s",
            binary_var_name="alpha"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add the L1 loss |x-x*|₁ as mixed-integer linear constraints.
        """
        if not torch.all(torch.from_numpy(self.system.x_lo_all) <
                         x_equilibrium) or\
                not torch.all(torch.from_numpy(self.system.x_up_all) >
                              x_equilibrium):
            raise Exception("add_state_error_l1_constraint: we currently " +
                            "require that x_lo < x_equilibrium < x_up")
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
                for j in range(Ain_x.shape[0]):
                    milp.addLConstr([torch.cat((
                        Ain_x[j].unsqueeze(0), Ain_s[j].unsqueeze(0),
                        Ain_alpha[j].unsqueeze(0)))],
                        [[x[i], s[i], alpha[i]]],
                        sense=gurobipy.GRB.LESS_EQUAL,
                        rhs=rhs_in[j] + Ain_x[j] * x_equilibrium[i])
        return (s, alpha)

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
        (z, beta, a_out, b_out) = self.add_relu_output_constraint(
            relu_model, relu_free_pattern, milp, x)

        # Now write the 1-norm |x - x*|₁ as mixed-integer linear constraints.
        # TODO(hongkai.di): support the case when x_equilibrium is not in the
        # strict interior of the state space.
        # Now compute ReLU(x*)
        (s, gamma) = self.add_state_error_l1_constraint(
            milp, x_equilibrium, x, slack_name="s", binary_var_name="gamma")
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

    def __init__(self, system):
        """
        @param system A AutonomousHybridLinearSystem instance.
        """
        super(LyapunovDiscreteTimeHybridSystem, self).__init__(system)

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
        (x, s, gamma, Aeq_s1, Aeq_gamma1) = self.add_hybrid_system_constraint(
            milp)
        # x_next is the variable x[n+1]
        x_next = milp.addVars(
            self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
            vtype=gurobipy.GRB.CONTINUOUS, name="x[n+1]")
        # Add the constraint x[n+1] = Aeq_s1 * s + Aeq_gamma1 * gamma
        for i in range(self.system.x_dim):
            milp.addLConstr(
                [torch.tensor([1.], dtype=milp.dtype), -Aeq_s1[i],
                 -Aeq_gamma1[i]], [[x_next[i]], s, gamma],
                sense=gurobipy.GRB.EQUAL, rhs=0.)

        # Add the mixed-integer constraint that formulates the output of
        # ReLU(x[n]).
        # z is the slack variable to write the output of ReLU(x[n]) with mixed
        # integer linear constraints.
        (z, beta, a_out, b_out) = self.add_relu_output_constraint(
            relu_model, relu_free_pattern, milp, x)

        # Now compute ReLU(x*)
        relu_x_equilibrium = relu_model.forward(x_equilibrium)

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
        (z_next, beta_next, _, _) = self.add_relu_output_constraint(
            relu_model, relu_free_pattern, milp, x_next)

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

    def __init__(self, system):
        super(LyapunovContinuousTimeHybridSystem, self).__init__(system)

    def add_relu_gradient_times_Aisi(
            self, relu_model, relu_free_pattern, milp, s, beta,
            Aisi_lower=None, Aisi_upper=None, slack_name="z"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add sum_i ∂ReLU(x)/∂x*Aᵢsᵢ as mixed-integer linear constraints.
        @param s The slack variable to write the hybrid linear dynamics as
        mixed-integer linear constraint. Returned from
        add_hybrid_system_constraint()
        @param beta The binary variable to determine the activation of the
        (leaky) ReLU units in the network, returned from
        add_relu_output_constraint()
        @param Aisi_lower Aisi_lower[i] is the lower bound of Aᵢsᵢ
        @param Aisi_upper Aisi_upper[i] is the lower bound of Aᵢsᵢ
        @return (z, a_out) z and a_out are both lists. z[i] are the slack
        variables to write ∂ReLU(x)/∂x*Aᵢsᵢ as mixed-integer linear constraint
        a_out[i].dot(z[i]) = ∂ReLU(x)/∂x*Aᵢsᵢ
        """
        assert(isinstance(
            relu_free_pattern, relu_to_optimization.ReLUFreePattern))
        assert(isinstance(milp, gurobi_torch_mip.GurobiTorchMIP))
        assert(isinstance(s, list))
        assert(isinstance(beta, list))

        z = [None] * self.system.num_modes
        a_out = [None] * self.system.num_modes
        if (Aisi_lower is None or Aisi_upper is None):
            Aisi_lower = [None] * self.system.num_modes
            Aisi_upper = [None] * self.system.num_modes
            for i in range(self.system.num_modes):
                Aisi_lower[i], Aisi_upper[i] = \
                    self.system.mode_derivative_bounds(i)
        else:
            assert(len(Aisi_lower) == self.system.num_modes)
            assert(len(Aisi_upper) == self.system.num_modes)
        for i in range(self.system.num_modes):
            # First write ∂ReLU(x)/∂x*Aᵢsᵢ
            a_out[i], A_Aisi, A_z, A_beta, rhs, _, _ = \
                relu_free_pattern.output_gradient_times_vector(
                    relu_model, torch.from_numpy(Aisi_lower[i]),
                    torch.from_numpy(Aisi_upper[i]))
            z[i] = milp.addVars(
                A_z.shape[1], lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name="z")
            A_si = A_Aisi @ self.system.A[i]
            for j in range(A_si.shape[0]):
                milp.addLConstr(
                    [A_si[j], A_z[j], A_beta[j]],
                    [s[i * self.system.x_dim:(i+1)*self.system.x_dim], z[i],
                     beta], sense=gurobipy.GRB.LESS_EQUAL, rhs=rhs[j],
                    name="milp_relu_gradient_times_Aisi")
        return (z, a_out)

    def add_relu_gradient_times_gigammai(
        self, relu_model, relu_free_pattern, milp, gamma, beta,
            gigammai_lower=None, gigammai_upper=None, slack_name="z"):
        """
        This function is intended for internal usage only (but I expose it
        as a public function for unit test).
        Add sum_i ∂ReLU(x)/∂x *gᵢγᵢ as mixed-integer linear constraints.
        @param gamma The binary variable indicating the active mode in the
        hybrid dynamical system. Returned from
        add_hybrid_system_constraint()
        @param beta The binary variable to determine the activation of the
        (leaky) ReLU units in the network, returned from
        add_relu_output_constraint()
        @param gigammai_lower gigammai_lower[i] is the lower bound of gᵢγᵢ
        @param gigammai_upper gigammai_upper[i] is the upper bound of gᵢγᵢ
        @return (z, a_out) z and a_out are both lists. z[i] are the slack
        variables to write ∂ReLU(x)/∂x*gᵢγᵢ as mixed-integer linear constraint
        a_out[i].dot(z[i]) = ∂ReLU(x)/∂x*gᵢγᵢ
        """
        assert(isinstance(
            relu_free_pattern, relu_to_optimization.ReLUFreePattern))
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
                relu_free_pattern.output_gradient_times_vector(
                    relu_model, gigammai_lower[i], gigammai_upper[i])
            z[i] = milp.addVars(
                A_z.shape[1], lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.CONTINUOUS, name=slack_name)
            A_gammai = A_gigammai @ self.system.g[i]
            for j in range(A_gammai.shape[0]):
                milp.addLConstr(
                    [A_gammai[j].unsqueeze(0), A_z[j], A_beta[j]],
                    [[gamma[i]], z[i], beta],
                    sense=gurobipy.GRB.LESS_EQUAL, rhs=rhs[j],
                    name="milp_relu_gradient_times_gigammai")
        return (z, a_out)

    def lyapunov_derivative_as_milp(
            self, relu_model, x_equilibrium, V_rho, epsilon,
            lyapunov_lower=None, lyapunov_upper=None):
        """
        We assume that the Lyapunov function
        V(x) = ReLU(x) - ReLU(x*) + ρ|x-x*|₁, where x* is the equilibrium
        state.
        Formulate the Lyapunov condition
        V̇(x) ≤ -ε V(x) for all x satisfying
        lower <= V(x) <= upper
        as the maximal of following optimization problem is no larger
        than 0.
        max V̇(x) + ε * V(x)
        s.t lower <= V(x) <= upper
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
        return (milp, x, s, gamma, z, beta)
        where milp is a GurobiTorchMILP object.
        The decision variables of the MILP are
        (x, s, gamma, z, beta)
        @param x_equilibrium The equilibrium state.
        @param V_rho ρ in the documentation above.
        @param epsilon The exponential convergence rate.
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

        (x, s, gamma, _, _) = self.add_hybrid_system_constraint(milp)

        # V̇ = ∂V/∂x(∑ᵢ Aᵢsᵢ + gᵢγᵢ)
        #   = ∑ᵢ(∂ReLU(x)/∂x*Aᵢsᵢ + ∂ReLU(x)/∂x*gᵢγᵢ
        #       + ρ*sign(x-x*) * Aᵢsᵢ + ρ*sign(x-x*)*gᵢγᵢ)
        # In order to compute ∂ReLU(x)/∂x*Aᵢsᵢ, ∂ReLU(x)/∂x*gᵢγᵢ, we first
        # introduce the binary variable β, which represents the activation of
        # each (leaky) ReLU unit in the network. Then we can call
        # output_gradient_times_vector().

        # We first get the mixed-integer linear constraint, which encode the
        # activation of beta and the network input.
        (relu_z1, relu_beta, a_relu_out, b_relu_out) = \
            self.add_relu_output_constraint(
                relu_model, relu_free_pattern, milp, x)

        # for each mode, we want to compute ∂V/∂x*Aᵢsᵢ, ∂V/∂x*gᵢγᵢ.
        # where ∂V/∂x=∂ReLU(x)/∂x + ρ*sign(x-x*)
        # In order to handle the part on ρ*sign(x-x*) * Aᵢsᵢ, ρ*sign(x-x*)*gᵢγᵢ
        # we first introduce binary variable α, such that
        # α(j) = 1 => x(j) - x*(j) >= 0
        # α(j) = 0 => x(j) - x*(j) <= 0
        # Hence sign(x(j) - x*(j)) = 2 * α - 1
        (s3, alpha) = self.add_state_error_l1_constraint(
                milp, x_equilibrium, x, slack_name="s3",
                binary_var_name="alpha")

        # z1[i] is the slack variable to write ∂ReLU(x)/∂x*Aᵢsᵢ as
        # mixed-integer linear constraints. cost_z1_coef is the coefficient of
        # z1 in the objective.
        Aisi_lower = [None] * self.system.num_modes
        Aisi_upper = [None] * self.system.num_modes
        for i in range(self.system.num_modes):
            Aisi_lower[i], Aisi_upper[i] =\
                self.system.mode_derivative_bounds(i)
        z1, cost_z1_coef = self.add_relu_gradient_times_Aisi(
            relu_model, relu_free_pattern, milp, s, relu_beta, Aisi_lower,
            Aisi_upper, slack_name="z1")
        gigammai_lower = [None] * self.system.num_modes
        gigammai_upper = [None] * self.system.num_modes
        for i in range(self.system.num_modes):
            gigammai_lower[i] = torch.min(
                torch.zeros(self.system.x_dim, dtype=self.system.dtype),
                self.system.g[i])
            gigammai_upper[i] = torch.max(
                torch.zeros(self.system.x_dim, dtype=self.system.dtype),
                self.system.g[i])
        z2, cost_z2_coef = self.add_relu_gradient_times_gigammai(
            relu_model, relu_free_pattern, milp, gamma, relu_beta,
            slack_name="z2")
        # z3[i] is the slack variable to write sign(x-x*)*Aᵢsᵢ as mixed-integer
        # linear constraints.
        z3 = [None] * self.system.num_modes
        # z4[i] is the slack variable to write sign(x-x*)*gᵢγᵢ as mixed-integer
        # linear constraints.
        z4 = [None] * self.system.num_modes
        # cost_z3_coef[i] is the coefficient of z3[i] in the cost function.
        cost_z3_coef = [None] * self.system.num_modes
        # cost_z4_coef[i] is the coefficient of z4[i] in the cost function.
        cost_z4_coef = [None] * self.system.num_modes
        # cost_s_coeff[i] is the coefficient of s[i] in the cost function.
        cost_s_coef = [None] * self.system.num_modes
        # cost_gamma_coeff[i] is the coefficient of gamma[i] in the cost
        # function.
        cost_gamma_coef = [None] * self.system.num_modes
        for i in range(self.system.num_modes):
            # Now handle the part on ρ*sign(x-x*) * Aᵢsᵢ, ρ*sign(x-x*)*gᵢγᵢ
            # We have the binary variable α, and sign(x-x*) = 2α-1
            # so ρ*sign(x-x*) * Aᵢsᵢ = ρ*2α*Aᵢsᵢ - ρ*Aᵢsᵢ
            # and ρ*sign(x-x*) * gᵢγᵢ = ρ*2α*gᵢγᵢ - ρ*gᵢγᵢ
            # z3[i] = α*Aᵢsᵢ
            z3[i] = milp.addVars(
                self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.BINARY, name="z3")
            z4[i] = milp.addVars(
                self.system.x_dim, lb=-gurobipy.GRB.INFINITY,
                vtype=gurobipy.GRB.BINARY, name="z4")
            for j in range(self.system.x_dim):
                Ain_Aisi_sign, Ain_z3, Ain_alpha1, rhs_in_sign1 = utils.\
                    replace_binary_continuous_product(
                        Aisi_lower[j], Aisi_upper[j])
                Ain_si_sign = Ain_Aisi_sign.reshape((-1, 1)) @ \
                    self.system.A[i][j].reshape((1, -1))
                for k in range(Ain_si_sign.shape[0]):
                    milp.addLConstr(
                        [Ain_si_sign[k], Ain_z3[k].unsqueeze(0),
                         Ain_alpha1[k].unsqueeze(0)],
                        [s[i * self.system.x_dim: (i+1) * self.system.x_dim],
                         [z3[i][j]], [alpha[i]]],
                        sense=gurobipy.GRB.LESS_EQUAL, rhs=rhs_in_sign1[k])
                Ain_gigammai_sign, Ain_z4, Ain_alpha2, rhs_in_sign2 = utils.\
                    replace_binary_continuous_product(
                        gigammai_lower[i][j], gigammai_upper[i][j])
                Ain_gammai_sign = Ain_gigammai_sign * self.system.g[i][j]
                for k in range(Ain_gammai_sign.shape[0]):
                    milp.addLConstr(
                        [Ain_gammai_sign[k].unsqueeze(0),
                         Ain_z4[k].unsqueeze(0), Ain_alpha2[k].unsqueeze(0)],
                        [[gamma[i]], [z4[i][j]], [alpha[i]]],
                        sense=gurobipy.GRB.LESS_EQUAL, rhs=rhs_in_sign2[k])

            cost_z3_coef[i] = 2 * V_rho * torch.ones(
                self.system.x_dim, dtype=self.system.dtype)
            cost_s_coef[i] =\
                -V_rho * torch.sum(self.system.A[i], dim=0)
            cost_z4_coef[i] = 2 * V_rho * torch.ones(
                self.system.x_dim, dtype=self.system.dtype)
            cost_gamma_coef[i] = \
                -V_rho * torch.sum(self.system.g[i]).unsqueeze(0)

        cost_s_coef_flat = torch.cat(cost_s_coef)
        cost_gamma_coef_flat = torch.cat(cost_gamma_coef)
        cost_coeffs = [mode_coef for coef_list in [
            cost_z1_coef, cost_z2_coef, cost_z3_coef, cost_z4_coef] for
            mode_coef in coef_list]
        cost_coeffs.append(cost_s_coef_flat)
        cost_coeffs.append(cost_gamma_coef_flat)
        cost_vars = [mode_var for var_list in [z1, z2, z3, z4] for mode_var
                     in var_list]
        cost_vars.append(s)
        cost_vars.append(gamma)

        cost_vars.append(relu_z1)
        cost_coeffs.append(a_relu_out * epsilon)

        cost_vars.append(s3)
        cost_coeffs.append(
            epsilon * V_rho * torch.ones(
                self.system.x_dim, dtype=self.system.dtype))
        # The cost is
        # max V̇ + εV
        #   = ∑ᵢ∂V/∂x(Aᵢsᵢ + gᵢγᵢ) + ε(ReLU(x) - ReLU(x*) + ρ|x-x*|₁)
        milp.setObjective(
            cost_coeffs, cost_vars,
            -epsilon * relu_model.forward(x_equilibrium).squeeze(),
            gurobipy.GRB.MAXIMIZE)

        return (milp, x)
