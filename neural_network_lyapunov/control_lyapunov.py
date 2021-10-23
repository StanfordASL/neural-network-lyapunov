import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.mip_utils as mip_utils
import torch
import numpy as np
import gurobipy


class ControlAffineSystemAddConstraintReturn(lyapunov.SystemConstraintReturn):
    """
    The return type of add_system_constraint()
    """
    def __init__(self, slack, binary):
        super(ControlAffineSystemAddConstraintReturn,
              self).__init__(slack, binary)
        self.mip_cnstr_f = None
        self.mip_cnstr_G = None
        self.f_lo = None
        self.f_up = None
        self.G_flat_lo = None
        self.G_flat_up = None


class LyapDerivMilpReturn:
    def __init__(self, milp, x, relu_beta, l1_binary, dl1dx_times_f,
                 system_constraint_return):
        """
        Args:
          relu_beta: The binary variable indicating the activeness of each ReLU
          unit in the network.
          dl1dx_times_f: refer to _add_dl1dx_times_f() for more details.
        """
        self.milp = milp
        self.x = x
        self.beta = relu_beta
        self.l1_binary = l1_binary
        self.dl1dx_times_f = dl1dx_times_f
        self.system_constraint_return = system_constraint_return


class SubgradientPolicy:
    """
    Since our Lyapunov function V(x) is a piecewise linear function of x, its
    gradient is not well-defined everywhere. When the gradient is not unique,
    we consider different policies to handle subgradient ∂V/∂x. For example
    1. Only consider the left and right derivatives of (leaky) ReLU and
       absolute value function.
    2. Consider a finite set of sampled subgradient.
    """
    def __init__(self, subgradient_samples=None):
        """
        The sampled subgradient. If set to None, then we only consider the left
        and right derivatives. Note that subgradient_samples should in the
        open-set of (left_derivative, right_derivative).
        """
        if subgradient_samples is None:
            self.subgradient_samples = None
        else:
            assert (isinstance(subgradient_samples, np.ndarray))
            self.subgradient_samples = subgradient_samples

    @property
    def search_subgradient(self):
        return self.subgradient_samples is not None


class ControlLyapunov(lyapunov.LyapunovHybridLinearSystem):
    """
    Given a control affine system with dynamics
    ẋ = f(x) + G(x)u
    with input bounds u_lo <= u <= u_up
    The conditions for its control Lyapunov function V(x) is
    V(x) > 0
    minᵤ V̇ < 0
    where we can compute minᵤ V̇ as
    minᵤ V̇
    = ∂V/∂x*f(x) + minᵤ ∂V/∂x*G(x)*u
    = ∂V/∂x*f(x) + ∂V/∂x*G(x)*(u_lo + u_up)/2
        - |∂V/∂x*G(x) * diag((u_up - u_lo)/2)|₁
    where |•|₁ denotes the 1-norm of a vector.

    The Lyapunov function V(x) is formulated as
    V(x) = ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁
    where ϕ is a neural network with (leaky) ReLU units.

    We will check if minᵤ V̇ < 0 is satisfied by solving a mixed-integer
    program with x being the decision variable.
    """
    def __init__(self,
                 system: control_affine_system.ControlPiecewiseAffineSystem,
                 lyapunov_relu, l1_subgradient_policy: SubgradientPolicy):
        """
        Args:
          system: A control-affine system.
          lyapunov_relu: The neural network ϕ which defines the Lyapunov
          function.
          l1_subgradient_policy: The policy for searching subgradient of
          absolute function in the l1-norm.
        """
        assert (isinstance(system,
                           control_affine_system.ControlPiecewiseAffineSystem))
        super(ControlLyapunov, self).__init__(system, lyapunov_relu)
        assert (isinstance(l1_subgradient_policy, SubgradientPolicy))
        self.l1_subgradient_policy = l1_subgradient_policy

    def lyapunov_derivative(self,
                            x,
                            x_equilibrium,
                            V_lambda,
                            epsilon,
                            *,
                            R,
                            subgradient_rule: str = "max_sample",
                            zero_tol: float = 0.):
        """
        Compute minᵤ V̇ + ε*V
        subject to u_lo <= u <= u_up
        Note that ∂V/∂x = ∂ϕ/∂x + λ ∑ᵢ sign(R[i, :](x−x*))R[i, :]

        Note that V is not differentiable everywhere. When V is not
        differentiable at x, we choose its sub-gradient according to
        subgradient_rule.

        Args:
          x_equilibrium: x* in the documentation.
          V_lambda: λ in the documentation.
          epsilon: ε in the documentation.
          R: A full column rank matrix. Use the identity matrix if R=None.
          subgradient_rule: Can be either "max_sample", "min_sample", or
            "all".
            If "max_sample", then we compute the maximal of V̇ among all
            sampled subgradients.
            If "min", then we compute the minimal of V̇ among all sampled
            subgradients.
            If "all", then we compute all V̇ for all sampled subgradients.
          zero_tol: The l1-norm and ReLU unit is not differentiable at 0.
          If the absolute value of an input to the l1-norm or ReLU unit is no
          larger than zero_tol, then we consider both the left and right
          derivatives.
        """
        assert (isinstance(x, torch.Tensor))
        assert (x.shape == (self.system.x_dim, ))

        R = lyapunov._get_R(R, self.system.x_dim, x_equilibrium.device)

        # First compute ∂ϕ/∂x
        dphi_dx = utils.relu_network_gradient(self.lyapunov_relu,
                                              x,
                                              zero_tol=zero_tol).squeeze(1)

        # Now compute the gradient of λ|R(x−x*)|₁
        dl1_dx = V_lambda * utils.l1_gradient(
            R @ (x - x_equilibrium),
            zero_tol=zero_tol,
            subgradient_samples=self.l1_subgradient_policy.subgradient_samples
        ) @ R

        # We compute the sum of each possible dphi_dX and dl1_dx
        dVdx = dphi_dx.repeat((dl1_dx.shape[0], 1)) + dl1_dx.repeat(
            (1, dphi_dx.shape[0])).view(
                (dphi_dx.shape[0] * dl1_dx.shape[0], self.system.x_dim))

        # minᵤ V̇
        # = ∂V/∂x*f(x) + ∂V/∂x*G(x)*(u_lo + u_up)/2
        #     - |∂V/∂x*G(x) * diag((u_up - u_lo)/2)|₁
        G = self.system.G(x)
        Vdot = dVdx @ self.system.f(x) + dVdx @ G @ (
            (self.system.u_lo + self.system.u_up) / 2) - torch.norm(
                (dVdx @ G) *
                ((self.system.u_up - self.system.u_lo) / 2).repeat(
                    (dVdx.shape[0], 1)),
                p=1,
                dim=1)
        V = self.lyapunov_value(x, x_equilibrium, V_lambda, R=R)
        if subgradient_rule == "max_sample":
            return torch.max(Vdot.squeeze()) + epsilon * V
        elif subgradient_rule == "min_sample":
            return torch.min(Vdot.squeeze()) + epsilon * V
        elif subgradient_rule == "all":
            return Vdot.squeeze() + epsilon * V
        else:
            raise Exception("lyapunov_derivative(): unknown subgradient_rule" +
                            f" {subgradient_rule}")

    def calc_vdot_allsubgradient(self,
                                 x,
                                 x_equilibrium,
                                 V_lambda,
                                 R,
                                 zero_tol: float = 0.):
        """
        Compute V̇ = minᵤ max_d dᵀ(f(x) + G(x)u)
        where d is the subgradient of V(x) at x.
        Notice this function only works when the set of subgradient at x
        is a convex set (namely if the neural network ϕ(x) has subgradient,
        then all the neurons with input 0 are on the same layer).

        This problem is formulated as the following optimization
        min_u s
        s.t s >= dᵢᵀ(f(x) + G(x)u)
        where dᵢ is the i'th vertex of the set of subgradient.
        """
        dphi_dx = utils.relu_network_gradient(self.lyapunov_relu,
                                              x,
                                              zero_tol=zero_tol)
        dl1_dx = utils.l1_gradient(R @ (x - x_equilibrium)) @ R
        f = self.system.f(x)
        G = self.system.G(x)
        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)
        u = milp.addVars(self.system.u_dim, lb=-gurobipy.GRB.INFINITY)
        milp.addMConstrs(
            [torch.eye(self.system.u_dim, dtype=self.system.dtype)], [u],
            sense=gurobipy.GRB.LESS_EQUAL,
            b=self.system.u_up)
        milp.addMConstrs(
            [-torch.eye(self.system.u_dim, dtype=self.system.dtype)], [u],
            sense=gurobipy.GRB.LESS_EQUAL,
            b=-self.system.u_lo)
        s = milp.addVars(1, lb=-gurobipy.GRB.INFINITY)
        for i in range(dphi_dx.shape[0]):
            for j in range(dl1_dx.shape[0]):
                V_subgradient = dphi_dx[i] + V_lambda * dl1_dx[j].unsqueeze(0)
                milp.addLConstr([
                    torch.tensor([1.], dtype=self.system.dtype),
                    (-V_subgradient @ G).squeeze(0)
                ], [s, u],
                                sense=gurobipy.GRB.GREATER_EQUAL,
                                rhs=V_subgradient.squeeze(0) @ f)
        milp.setObjective([torch.tensor([1.], dtype=self.system.dtype)], [s],
                          constant=0.,
                          sense=gurobipy.GRB.MINIMIZE)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.optimize()
        assert (milp.gurobi_model.status == gurobipy.GRB.Status.OPTIMAL)
        return torch.tensor(milp.gurobi_model.ObjVal,
                            dtype=self.system.dtype), np.array(
                                [v.x for v in u])
        # return milp.compute_objective_from_mip_data_and_solution()

    def add_system_constraint(
        self,
        milp: gurobi_torch_mip.GurobiTorchMIP,
        x: list,
        f: list,
        Gt: list,
        *,
        binary_var_type=gurobipy.GRB.BINARY
    ) -> ControlAffineSystemAddConstraintReturn:
        """
        Add the (mixed-integer linear) constraints of f(x) and G(x).

        Args:
          Gt: Gt is a 2D list. len(G) = u_dim. len(G[i]) = x_dim. Gt is the
          transpose of system.G(x).
        """
        mip_cnstr_ret, slack_f, slack_G, binary_f, binary_G = \
            control_affine_system.add_system_constraint(
                self.system, milp, x, f, Gt, binary_var_type=binary_var_type)
        ret = ControlAffineSystemAddConstraintReturn(slack_f + slack_G,
                                                     binary_f + binary_G)
        ret.mip_cnstr_f = mip_cnstr_ret.mip_cnstr_f
        ret.mip_cnstr_G = mip_cnstr_ret.mip_cnstr_G
        ret.f_lo = mip_cnstr_ret.f_lo
        ret.f_up = mip_cnstr_ret.f_up
        ret.G_flat_lo = mip_cnstr_ret.G_flat_lo
        ret.G_flat_up = mip_cnstr_ret.G_flat_up
        return ret

    def lyapunov_derivative_as_milp(
            self,
            x_equilibrium,
            V_lambda,
            epsilon,
            eps_type: lyapunov.ConvergenceEps,
            *,
            R,
            lyapunov_lower=None,
            lyapunov_upper=None,
            x_warmstart=None,
            binary_var_type=gurobipy.GRB.BINARY) -> LyapDerivMilpReturn:
        """
        Formulate maxₓ f(x)
                  s.t lyapunov_lower <= V(x) <= lyapunov_upper
                      x_lo <= x <= x_up
        as a mixed-integer linear program.
        The objective f(x) takes different forms depending on @p eps_type.
        If eps_type = ConvergenceEps.kExpLower:
          f(x) =  V̇ + εV(x)
        if eps_type = ConvergenceEps.kExpUpper:
          f(x) =  -V̇ - εV(x)
        if eps_type = ConvergenceEps.kAsymp:
          f(x) = V̇ + ε|R(x−x*)|₁
        The lyapunov function is defined as V(x) = ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁

        Args:
          x_equilibrium: The equilibrium state x*
          V_lambda: λ in the Lyapunov function.
          epsilon: ε in the documentation above.
          eps_type: Refer to ConvergenceEps for more details.
          R: A full column-rank matrix.
          lyapunov_lower: The lower bound of the Lyapunov function.
          lyapunov_upper: The upper bound of the Lyapunov function.
          x_warmstart: Warm-start value for x in the optimization.
          binary_var_type: whether to register the binary variable as
          continuous relaxation or not.

        Returns:
          (milp, x, beta, gamma, s, z)
        """
        assert (isinstance(x_equilibrium, torch.Tensor))
        assert (x_equilibrium.shape == (self.system.x_dim, ))
        if lyapunov_lower is not None:
            assert (isinstance(lyapunov_lower, float))
        if lyapunov_upper is not None:
            assert (isinstance(lyapunov_upper, float))
        assert (isinstance(V_lambda, float))
        assert (isinstance(epsilon, float))
        assert (isinstance(eps_type, lyapunov.ConvergenceEps))
        R = lyapunov._get_R(R, self.system.x_dim, x_equilibrium.device)

        # First add the system dynamics constraint f(x), G(x).
        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)
        x = milp.addVars(self.system.x_dim,
                         lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="x")
        f = milp.addVars(self.system.x_dim,
                         lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="f")
        # Notice that torch.tensor(Gt) is the transpose of system.G(x)
        Gt = [None] * self.system.u_dim
        for i in range(self.system.u_dim):
            Gt[i] = milp.addVars(self.system.x_dim,
                                 lb=-gurobipy.GRB.INFINITY,
                                 vtype=gurobipy.GRB.CONTINUOUS,
                                 name=f"Gt[{i}]")

        system_constraint_return = self.add_system_constraint(
            milp, x, f, Gt, binary_var_type=binary_var_type)

        # We need to compute ∂ϕ/∂x*f(x) and ∂ϕ/∂x*G(x), which requires the
        # binary variables indicating the activeness of each ReLU unit.
        relu_slack, relu_beta, a_relu_out, b_relu_out, _ = \
            self.add_lyap_relu_output_constraint(milp, x)

        mip_cnstr_dphidx_times_f = \
            self.lyapunov_relu_free_pattern.output_gradient_times_vector(
                system_constraint_return.f_lo, system_constraint_return.f_up)

        dphidx_times_f_slack, _ = milp.add_mixed_integer_linear_constraints(
            mip_cnstr_dphidx_times_f, f, None, "f_slack", relu_beta, "f_ineq",
            "f_eq", "f", binary_var_type)
        # I expect these entries to be None, so that Vdot_coeff/Vdot_vars
        # won't include these entries.
        assert (mip_cnstr_dphidx_times_f.Aout_input is None)
        assert (mip_cnstr_dphidx_times_f.Aout_binary is None)
        assert (mip_cnstr_dphidx_times_f.Cout is None)
        # V̇ will be Vdot_coeff * Vdot_vars + Vdot_constant. We will use V̇ in
        # the objective.
        Vdot_coeff = [mip_cnstr_dphidx_times_f.Aout_slack.reshape((-1, ))]
        # Add ∂ϕ/∂x*f to V̇
        Vdot_vars = [dphidx_times_f_slack]
        Vdot_constant = torch.tensor(0, dtype=self.system.dtype)

        # We need to compute |R(x−x*)|₁
        l1_slack, l1_binary = self.add_state_error_l1_constraint(
            milp,
            x_equilibrium,
            x,
            R=R,
            slack_name="l1_slack",
            binary_var_name="l1_binary",
            binary_var_type=binary_var_type,
            binary_for_zero_input=self.l1_subgradient_policy.search_subgradient
        )
        l1_subgradient_binary = self._add_l1_subgradient_binary(
            milp, l1_binary)
        if (self.network_bound_propagate_method ==
                mip_utils.PropagateBoundsMethod.IA):
            Rf_lo, Rf_up = self._compute_Rf_bounds_IA(
                R, system_constraint_return.f_lo,
                system_constraint_return.f_up)
            RG_lo, RG_up = self._compute_RG_bounds_IA(
                R, system_constraint_return.G_flat_lo,
                system_constraint_return.G_flat_up)
        else:
            raise NotImplementedError
        dl1dx_times_f = self._add_dl1dx_times_f(milp, x, l1_binary,
                                                l1_subgradient_binary, f, R,
                                                Rf_lo, Rf_up, V_lambda,
                                                Vdot_coeff, Vdot_vars)
        dVdx_times_G_ret, dVdx_times_G_binary = self._add_dVdx_times_G(
            milp, x, l1_binary, relu_beta, l1_subgradient_binary, Gt, R,
            system_constraint_return.G_flat_lo,
            system_constraint_return.G_flat_up, RG_lo, RG_up, V_lambda,
            Vdot_coeff, Vdot_vars)
        if eps_type in (lyapunov.ConvergenceEps.ExpLower,
                        lyapunov.ConvergenceEps.ExpUpper):
            # The cost is V̇ + ε*V= V̇ + ε(ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁)
            objective_coeff = Vdot_coeff + [epsilon * a_relu_out] + [
                epsilon * V_lambda *
                torch.ones(R.shape[0], dtype=self.system.dtype)
            ]
            objective_vars = Vdot_vars + [relu_slack] + [l1_slack]
            objective_constant = Vdot_constant + epsilon * b_relu_out - \
                epsilon * self.lyapunov_relu(x_equilibrium).squeeze()
            if eps_type == lyapunov.ConvergenceEps.ExpUpper:
                # negate the cost.
                objective_coeff = [-coeff for coeff in objective_coeff]
                objective_constant *= -1
        elif eps_type == lyapunov.ConvergenceEps.Asymp:
            # The cost is V̇ + ε |R(x−x*)|₁
            objective_coeff = Vdot_coeff + [
                epsilon * torch.ones(R.shape[0], dtype=self.system.dtype)
            ]
            objective_vars = Vdot_vars + [l1_slack]
            objective_constant = Vdot_constant

        milp.setObjective(objective_coeff,
                          objective_vars,
                          objective_constant,
                          sense=gurobipy.GRB.MAXIMIZE)

        return LyapDerivMilpReturn(milp, x, relu_beta, l1_binary,
                                   dl1dx_times_f, system_constraint_return)

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
        Sample states xⁱ, i=1,...N with loss l(xⁱ) for each sample. We penalize
        the total loss as reduction(max(l(xⁱ) + margin, 0))
        If eps_type = ExpLower, then l(x) = V̇(x) + εV(x)
        If eps_type = ExpUpper, then l(x) = -V̇(x) - εV(x)
        If eps_type = Asymp, then l(x) = V̇(x) + ε|R(x−x*)|₁

        If @p reduction = "mean", then reduction(x) = mean(x)
        If @p reduction = "max", then reduction(x) = max(x)
        If @p reduction = "4norm", then reduction(x) = |x|₄

        Args:
          state_next: This is a dummy variable for API consistency. It has to
          be None in this function, and we won't use it.
          weight: Has to be None. We use uniform weight for each sample.
        """
        assert (isinstance(V_lambda, float))
        assert (isinstance(epsilon, float))
        assert (isinstance(state_samples, torch.Tensor))
        assert (state_samples.shape[1] == self.system.x_dim)
        assert (state_next is None)
        assert (reduction in {"mean", "max", "4norm"})
        assert (weight is None)
        R = lyapunov._get_R(R, self.system.x_dim, state_samples.device)
        if eps_type in (lyapunov.ConvergenceEps.ExpLower,
                        lyapunov.ConvergenceEps.Asymp):
            subgradient_rule = "max_sample"
        elif eps_type == lyapunov.ConvergenceEps.ExpUpper:
            subgradient_rule = "min_sample"

        Vdot_samples = torch.stack([
            self.lyapunov_derivative(state_samples[i],
                                     x_equilibrium,
                                     V_lambda,
                                     epsilon,
                                     R=R,
                                     subgradient_rule=subgradient_rule,
                                     zero_tol=1E-6)
            for i in range(state_samples.shape[0])
        ])
        V_samples = self.lyapunov_value(state_samples,
                                        x_equilibrium,
                                        V_lambda,
                                        R=R).squeeze()
        if eps_type == lyapunov.ConvergenceEps.ExpLower:
            hinge_loss_all = torch.nn.HingeEmbeddingLoss(
                margin=margin,
                reduction="none")(-(Vdot_samples + epsilon * V_samples),
                                  torch.tensor(-1.).to(state_samples.device))
        elif eps_type == lyapunov.ConvergenceEps.ExpUpper:
            hinge_loss_all = torch.nn.HingeEmbeddingLoss(
                margin=margin,
                reduction="none")((Vdot_samples + epsilon * V_samples),
                                  torch.tensor(-1.).to(state_samples.device))
        elif eps_type == lyapunov.ConvergenceEps.Asymp:
            hinge_loss_all = torch.nn.HingeEmbeddingLoss(
                margin=margin,
                reduction="none")(-(Vdot_samples + epsilon * torch.norm(
                    R @ (state_samples - x_equilibrium).T, p=1, dim=0)),
                                  torch.tensor(-1.).to(state_samples.device))
        if reduction == "mean":
            return torch.mean(hinge_loss_all)
        elif reduction == "max":
            return torch.max(hinge_loss_all)
        elif reduction == "4norm":
            return torch.norm(hinge_loss_all, p=4)

    def _add_l1_subgradient_binary(self, milp, l1_binary):
        if self.l1_subgradient_policy.search_subgradient:
            l1_subgradient_binary = [None] * len(l1_binary)
            for i in range(len(l1_binary)):
                l1_subgradient_binary[i] = milp.addVars(
                    self.l1_subgradient_policy.subgradient_samples.size,
                    vtype=gurobipy.GRB.BINARY,
                    name=f"subgradient_binary[{i}]")
                # Add the constraint
                # sum_j subgradient_binary[i][j] = l1_binary[i][1]
                milp.addLConstr([
                    torch.ones(
                        self.l1_subgradient_policy.subgradient_samples.size,
                        dtype=self.system.dtype),
                    torch.tensor([-1], dtype=self.system.dtype)
                ], [l1_subgradient_binary[i], [l1_binary[i][1]]],
                                sense=gurobipy.GRB.EQUAL,
                                rhs=0.)
        else:
            l1_subgradient_binary = None
        return l1_subgradient_binary

    def _compute_Rf_bounds_IA(self, R, f_lo, f_up):
        """
        Compute the range of R * f.
        """
        return mip_utils.compute_range_by_IA(
            R, torch.zeros(R.shape[0], dtype=self.system.dtype), f_lo, f_up)

    def _compute_RG_bounds_IA(self, R, G_flat_lo, G_flat_up):
        """
        Compute the range of R * G
        """
        RG_lo = torch.empty((R.shape[0], self.system.u_dim),
                            dtype=self.system.dtype)
        RG_up = torch.empty((R.shape[0], self.system.u_dim),
                            dtype=self.system.dtype)
        G_lo = G_flat_lo.reshape((self.system.x_dim, self.system.u_dim))
        G_up = G_flat_up.reshape((self.system.x_dim, self.system.u_dim))
        for i in range(self.system.u_dim):
            RG_lo[:, i], RG_up[:, i] = mip_utils.compute_range_by_IA(
                R, torch.zeros(R.shape[0], dtype=self.system.dtype),
                G_lo[:, i], G_up[:, i])
        return RG_lo, RG_up

    def _add_dl1dx_times_f(self, milp: gurobi_torch_mip.GurobiTorchMIP,
                           x: list, l1_binary: list, subgradient_binary: list,
                           f: list, R: torch.Tensor, Rf_lo: torch.Tensor,
                           Rf_up: torch.Tensor, V_lambda: float,
                           Vdot_coeff: list, Vdot_vars: list) -> list:
        """
        Adds λ*∂|R(x−x*)|₁/∂x * f to V̇.
        We need to add the mixed-integer linear constraints, and also append
        new terms to Vdot_coeff and Vdot_vars.
        As we have introduced binary variables α for the l1 norm, such that
        If search_subgradient=True:
        α[i][0] = 1 => (R(x-x*))[i] <= 0
        α[i][1] = 1 => (R(x-x*))[i] = 0
        α[i][2] = 1 => (R(x-x*))[i] >= 0
        We know that λ * ∂|R(x−x*)|₁/∂x * f =
        λ * (-α[:][0] + α[:][2])ᵀ * R * f + λ * α[:][1]ᵀ * t
        where -|(R*f)[i]| <= t <= |(R*f)[i]|

        If search_subgradient=False
        α[i] = 1 => (R(x-x*))[i] >= 0
        α[i] = 0 => (R(x-x*))[i] <= 0
        we know that ∂|R(x−x*)|₁/∂x = (2α-1)ᵀ * R. Hence
        λ*∂|R(x−x*)|₁/∂x * f = 2λ*αᵀ*R*f − λ*1ᵀ*R*f

        Args:
          l1_binary: α in the documentation above.
          f: The variables representing the dynamics term f
          Rf_lo, Rf_up: The lower/upper bounds of R*f

        Return:
          dl1dx_times_f: The newly added variable representing λ∂|R(x−x*)|/∂x*f
        """
        # We need to add slack variables to represent the product between the
        # binary variable α and R*f
        assert (len(l1_binary) == R.shape[0])
        if self.l1_subgradient_policy.search_subgradient:
            dl1dx_times_f = _compute_dl1dx_times_y_sampled_subgradient(
                milp, l1_binary, f, subgradient_binary,
                self.l1_subgradient_policy.subgradient_samples, R, Rf_lo,
                Rf_up, V_lambda, "dl1dx_times_f")
        else:
            dl1dx_times_f = _compute_dl1dx_times_y(milp, l1_binary, f, R,
                                                   Rf_lo, Rf_up, V_lambda,
                                                   "dl1dx_times_f", False)
        Vdot_coeff.append(torch.tensor([1], dtype=self.system.dtype))
        Vdot_vars.append([dl1dx_times_f])
        return dl1dx_times_f

    def _add_dVdx_times_G(self, milp: gurobi_torch_mip.GurobiTorchMIP, x: list,
                          l1_binary: list, relu_beta: list,
                          l1_subgradient_binary: list, Gt: list,
                          R: torch.Tensor, G_flat_lo: torch.Tensor,
                          G_flat_up: torch.Tensor, RG_lo: torch.Tensor,
                          RG_up: torch.Tensor, V_lambda: float,
                          Vdot_coeff: list, Vdot_vars: list):
        """
        Add min_u ∂V/∂x * G * u
            s.t u_lo <= u <= u_up
        to V̇
        Note that this minimization problem can be written in the closed form
        as ∂V/∂x * G * (u_lo + u_up)/2
            - |∑ᵢ∂V/∂x * G.col(i) * (u_up(i) - u_lo(i))/2|₁
        where ∂V/∂x = ∂ϕ/∂x + λ*∂|R(x−x*)|₁/∂x

        Args:
          l1_binary: The binary variable to indicate the sign of the 1-norm
          |R(x−x*)|₁
          Gt: The variables representing the dynamics term G.transpose()
          G_flat_lo, G_flat_up: The lower and upper bounds of G.reshape((-1,))
          RG_lo, RG_up: The lower/upper bounds of R*G

        Return:
          dVdx_times_G_return: Refer to ComputeDvdxTimesGReturn
          dVdx_times_G_binary: The binary variable indicating the sign of
          ∂V/∂x * G
        """
        assert (len(l1_binary) == R.shape[0])
        assert (G_flat_lo.shape == (self.system.x_dim * self.system.u_dim, ))
        assert (G_flat_up.shape == (self.system.x_dim * self.system.u_dim, ))
        assert (RG_lo.shape == (R.shape[0], self.system.u_dim))
        assert (RG_up.shape == (R.shape[0], self.system.u_dim))
        compute_dVdx_times_G_return = self._compute_dVdx_times_G(
            milp, x, relu_beta, l1_binary, l1_subgradient_binary, Gt,
            G_flat_lo, G_flat_up, RG_lo, RG_up, R, V_lambda)

        dVdx_times_G_binary = milp.addVars(self.system.u_dim,
                                           vtype=gurobipy.GRB.BINARY,
                                           name="dVdx_times_G_binary")
        for i in range(self.system.u_dim):
            # We add slack and binary variables for the absolute value
            # |∂V/∂x * G.col(i)|
            # We add -|∂V/∂x * G.col(i)| * (u_up(i) - u_lo(i))/2
            if compute_dVdx_times_G_return.dVdx_times_G_lo[i] >= 0:
                # The binary variable equals to 1.
                milp.addLConstr([torch.tensor([1], dtype=self.system.dtype)],
                                [[dVdx_times_G_binary[i]]],
                                sense=gurobipy.GRB.EQUAL,
                                rhs=1.)
                dVdx_times_G_binary[i].lb = 1.
                dVdx_times_G_binary[i].ub = 1.
                # Add the cost (∂V/∂x * G)[i] * u_lo(i)
                Vdot_coeff.append(self.system.u_lo[i].reshape((1, )))
                Vdot_vars.append([compute_dVdx_times_G_return.dVdx_times_G[i]])
            elif compute_dVdx_times_G_return.dVdx_times_G_up[i] <= 0:
                # The binary variable equals to 0.
                milp.addLConstr([torch.tensor([1], dtype=self.system.dtype)],
                                [[dVdx_times_G_binary[i]]],
                                sense=gurobipy.GRB.EQUAL,
                                rhs=0.)
                dVdx_times_G_binary[i].lb = 0.
                dVdx_times_G_binary[i].ub = 0.
                # Add the cost (∂V/∂x * G)[i] * u_up(i)
                Vdot_coeff.append(self.system.u_up[i].reshape((1, )))
                Vdot_vars.append([compute_dVdx_times_G_return.dVdx_times_G[i]])
            else:
                mip_cnstr_abs = \
                    utils.absolute_value_as_mixed_integer_constraint(
                        compute_dVdx_times_G_return.dVdx_times_G_lo[i],
                        compute_dVdx_times_G_return.dVdx_times_G_up[i],
                        binary_for_zero_input=False)
                slack, _ = milp.add_mixed_integer_linear_constraints(
                    mip_cnstr_abs,
                    [compute_dVdx_times_G_return.dVdx_times_G[i]], None,
                    f"dVdx_times_G[{i}]_abs", [dVdx_times_G_binary[i]], "", "",
                    "")
                # Add the cost ∂V/∂x*G(x)[i] * (u_lo[i] + u_up[i])/2
                # - |∂V/∂x*G(x)[i]| * (u_up[i] - u_lo[i])/2
                Vdot_coeff.append(
                    torch.stack(
                        ((self.system.u_lo[i] + self.system.u_up[i]) / 2,
                         -(self.system.u_up[i] - self.system.u_lo[i]) / 2)))
                Vdot_vars.append(
                    [compute_dVdx_times_G_return.dVdx_times_G[i]] + slack)

        return compute_dVdx_times_G_return, dVdx_times_G_binary

    def validate_x_equilibrium(self, x_equilibrium: torch.Tensor):
        super(ControlLyapunov, self).validate_x_equilibrium(x_equilibrium)
        assert (self.system.can_be_equilibrium_state(x_equilibrium))

    class ComputeDvdxTimesGReturn:
        def __init__(self, dVdx_times_G, dVdx_times_G_lo, dVdx_times_G_up,
                     dphidx_times_G, dl1dx_times_G):
            """
            dphidx_times_G: ∂ϕ/∂x*G(x)
            dl1dx_times_G: λ∂|R(x−x*)|/∂x*G
            """
            self.dVdx_times_G = dVdx_times_G
            self.dVdx_times_G_lo = dVdx_times_G_lo
            self.dVdx_times_G_up = dVdx_times_G_up
            self.dphidx_times_G = dphidx_times_G
            self.dl1dx_times_G = dl1dx_times_G

    def _compute_dVdx_times_G(self, milp: gurobi_torch_mip.GurobiTorchMIP,
                              x: list, relu_beta: list, l1_binary: list,
                              l1_subgradient_binary: list, Gt: list,
                              G_flat_lo: torch.Tensor, G_flat_up: torch.Tensor,
                              RG_lo: torch.Tensor, RG_up: torch.Tensor,
                              R: torch.Tensor, V_lambda: float):
        """
        Add the constraint for ∂V/∂x * G(x) to the program.
        Namely we introduce a new variable dVdx_times_G, add the constraint
        dVdx_times_G = ∂V/∂x * G(x)
        Note that ∂V/∂x * G(x) = ∂ϕ/∂x*G(x) + λ*∂|R(x−x*)|₁/∂x*G(x)
        We have binary variable α for the l1 norm, such that
        α = 1 => (R(x-x*))(i) >= 0
        α = 0 => (R(x-x*))(i) <= 0
        we know that ∂|R(x−x*)|₁/∂x = (2α-1)ᵀ * R. Hence
        λ*∂|R(x−x*)|₁/∂x * G.col(i) = 2λ*αᵀ*R*G.col(i) − λ*1ᵀ*R*G.col(i)

        Args:
          relu_beta: The activation of each ReLU unit for this x in the network
          ϕ
          l1_binary: The activation of the l1-norm |R(x−x*)|₁

        Return:
        """
        dtype = self.system.dtype
        dVdx_times_G = milp.addVars(self.system.u_dim,
                                    lb=-gurobipy.GRB.INFINITY,
                                    name="dVdx_times_G")
        # The mip constraints for ∂ϕ/∂x*G(x).col(i).
        mip_cnstr_dphidx_times_G = [None] * self.system.u_dim
        if self.network_bound_propagate_method == \
                mip_utils.PropagateBoundsMethod.IA:
            for j in range(self.system.u_dim):
                Gi_lo = torch.stack([
                    G_flat_lo[i * self.system.u_dim + j]
                    for i in range(self.system.x_dim)
                ])
                Gi_up = torch.stack([
                    G_flat_up[i * self.system.u_dim + j]
                    for i in range(self.system.x_dim)
                ])
                mip_cnstr_dphidx_times_G[j] = self.lyapunov_relu_free_pattern.\
                    output_gradient_times_vector(Gi_lo, Gi_up)
        else:
            raise NotImplementedError
        # An alternative formulation is not to introduce the variable
        # dphidx_times_G, but write it as an affine function of the slack
        # variable in mip_cnstr_dphidx_times_G.
        dphidx_times_G = milp.addVars(self.system.u_dim,
                                      lb=-gurobipy.GRB.INFINITY,
                                      name="dphidx_times_G")
        for i in range(self.system.u_dim):
            milp.add_mixed_integer_linear_constraints(
                mip_cnstr_dphidx_times_G[i], Gt[i], [dphidx_times_G[i]],
                f"dphidx_times_G[{i}]_slack", relu_beta,
                f"dphidx_times_G[{i}]_ineq", f"dphidx_times_G[{i}]_eq",
                f"G[{i}]")

        # dl1dx_times_G[i] = λ∂|R(x−x*)|/∂x*G.col(i)
        dl1dx_times_G = [None] * self.system.u_dim
        for i in range(self.system.u_dim):
            if self.l1_subgradient_policy.search_subgradient:
                dl1dx_times_G[i] = _compute_dl1dx_times_y_sampled_subgradient(
                    milp, l1_binary, Gt[i], l1_subgradient_binary,
                    self.l1_subgradient_policy.subgradient_samples, R,
                    RG_lo[:, i], RG_up[:, i], V_lambda, f"dl1dx_times_G[{i}]")
            else:
                dl1dx_times_G[i] = _compute_dl1dx_times_y(
                    milp, l1_binary, Gt[i], R, RG_lo[:, i], RG_up[:, i],
                    V_lambda, f"dl1dx_times_G[{i}]", False)

        # ∂V/∂x * G(x) = ∂ϕ/∂x*G(x) + λ*∂|R(x−x*)|₁/∂x*G(x)
        # Hence we add the constraint
        # ∂ϕ/∂x*G + λ*∂|R(x−x*)|₁/∂x*G(x)- ∂V/∂x * G = 0
        for i in range(self.system.u_dim):
            milp.addLConstr([
                torch.tensor([1, 1, -1], dtype=dtype),
            ], [[dphidx_times_G[i], dl1dx_times_G[i], dVdx_times_G[i]]],
                            sense=gurobipy.GRB.EQUAL,
                            rhs=0.)
        # Now compute the bounds on dVdx_times_G
        dVdx_times_G_lo = torch.empty(self.system.u_dim, dtype=dtype)
        dVdx_times_G_up = torch.empty(self.system.u_dim, dtype=dtype)
        # ∂|R(x−x*)|₁/∂x*G(x).col(i) is in the range of [-bnd, bnd], where
        # bnd = [∑ⱼ(max(abs(RG_lo[j, i]), abs(RG_up[j, i])))
        for i in range(self.system.u_dim):
            bnd = torch.sum(
                torch.maximum(torch.abs(RG_lo[:, i]), torch.abs(RG_up[:, i])))
            dVdx_times_G_lo[
                i] = mip_cnstr_dphidx_times_G[i].Wz_lo[-1] + V_lambda * -bnd
            dVdx_times_G_up[
                i] = mip_cnstr_dphidx_times_G[i].Wz_up[-1] + V_lambda * bnd
        return ControlLyapunov.ComputeDvdxTimesGReturn(dVdx_times_G,
                                                       dVdx_times_G_lo,
                                                       dVdx_times_G_up,
                                                       dphidx_times_G,
                                                       dl1dx_times_G)


def _compute_dl1dx_times_y(milp: gurobi_torch_mip.GurobiTorchMIP,
                           l1_binary: list, y: list, R: torch.Tensor,
                           Ry_lo: torch.Tensor, Ry_up: torch.Tensor,
                           V_lambda: float, return_var_name: str,
                           search_subgradient: bool) -> gurobipy.Var:
    """
    Adds the constraint z = λ*∂|R(x−x*)|₁/∂x * y
    We need to add this as mixed-integer linear constraints, and also
    return z.

    If we don't search for subgradient, but just the left and right
    derivatives, then l1_binary is a list of binary variables α, meaning
    α[i] = 0 => (R*(x-x*))[i] <= 0
    α[i] = 1 => (R*(x-x*))[i] >= 0
    and ∂|R(x−x*)|₁/∂x = (2α-1)ᵀ*R
    hence z = 2λ*αᵀ*R*y − λ*1ᵀ*R*y

    If we search for subgradient, then l1_binary is a list of size-3 lists,
    with meaning
    α[i][0] = 1 => (R*(x-x*))[i] <= 0
    α[i][1] = 1 => (R*(x-x*))[i] = 0
    α[i][2] = 1 => (R*(x-x*))[i] >= 0
    ∂|R(x−x*)|₁/∂x * y = -α[:][0]ᵀ*R*y + α[:][2]ᵀ*R*y + α[:][1]ᵀ*t
    where -|(R * y)[i]| <= t[i] <= |(R * y)[i]|

    Args:
      y: The list of variables y.
      Ry_lo: The lower bound of R * y.
      Ry_up: The upper bound of R * y.
    """
    assert (len(l1_binary) == R.shape[0])
    assert (len(y) == R.shape[1])
    dtype = R.dtype
    dl1dx_times_y_var = milp.addVars(1,
                                     lb=-gurobipy.GRB.INFINITY,
                                     name=return_var_name)
    if not search_subgradient:
        # Only search for the left and right derivatives.
        # slack[i] = α[i] * R.row(i) * y
        slack = milp.addVars(len(l1_binary),
                             lb=-gurobipy.GRB.INFINITY,
                             name="l1_binary_times_Ry")
        for i in range(len(l1_binary)):
            A_Ry, A_slack, A_alpha, rhs = \
                utils.replace_binary_continuous_product(
                    Ry_lo[i], Ry_up[i], dtype=dtype)
            A_y = A_Ry.reshape((-1, 1)) @ R[i, :].reshape((1, -1))
            milp.addMConstrs(
                [A_y, A_slack.reshape((-1, 1)),
                 A_alpha.reshape((-1, 1))], [y, [slack[i]], [l1_binary[i]]],
                sense=gurobipy.GRB.LESS_EQUAL,
                b=rhs)
        milp.addLConstr([
            2 * V_lambda * torch.ones(len(l1_binary), dtype=dtype),
            -V_lambda * torch.ones(R.shape[0], dtype=dtype) @ R,
            -torch.tensor([1], dtype=dtype)
        ], [slack, y, dl1dx_times_y_var],
                        sense=gurobipy.GRB.EQUAL,
                        rhs=0.)
    else:
        alpha0_times_Ry, alpha2_times_Ry = _compute_alpha_times_Ry(
            milp, y, l1_binary, R, Ry_lo, Ry_up)
        # t is the slack variable for the subgradient at 0 times R * y.
        t = milp.addVars(R.shape[0], lb=-gurobipy.GRB.INFINITY)
        # alpha1_times_t[i] = α[i][1] * t[i]
        alpha1_times_t = milp.addVars(R.shape[0], lb=-gurobipy.GRB.INFINITY)
        for i in range(R.shape[0]):
            if Ry_lo[i] >= 0:
                # Add the constraint -R.row(i)*y <= t[i] <= R.row(i)*y
                milp.addLConstr([torch.tensor([1], dtype=dtype), -R[i, :]],
                                [[t[i]], y],
                                sense=gurobipy.GRB.LESS_EQUAL,
                                rhs=0.)
                milp.addLConstr([torch.tensor([-1], dtype=dtype), -R[i, :]],
                                [[t[i]], y],
                                sense=gurobipy.GRB.LESS_EQUAL,
                                rhs=0.)
                ti_lo = -Ry_up[i]
                ti_up = Ry_up[i]
            elif Ry_up[i] <= 0:
                # Add the constraint R.row(i)*y <= t[i] <= -R.row(i)*y
                milp.addLConstr([torch.tensor([-1], dtype=dtype), R[i, :]],
                                [[t[i]], y],
                                sense=gurobipy.GRB.LESS_EQUAL,
                                rhs=0.)
                milp.addLConstr([torch.tensor([1], dtype=dtype), R[i, :]],
                                [[t[i]], y],
                                sense=gurobipy.GRB.LESS_EQUAL,
                                rhs=0.)
                ti_lo = Ry_lo[i]
                ti_up = -Ry_lo[i]
            else:
                # We need to add a slack variable and binary variable to write
                # the absolute value |R.row(i)*y|.
                mip_cnstr_abs = \
                    utils.absolute_value_as_mixed_integer_constraint(
                        Ry_lo[i], Ry_up[i], binary_for_zero_input=False)
                mip_cnstr_abs.transform_input(R[i, :].reshape((1, -1)),
                                              torch.tensor([0], dtype=dtype))
                Ry_abs, Ry_abs_binary = \
                    milp.add_mixed_integer_linear_constraints(
                        mip_cnstr_abs, y, None, "", "", "", "", "")
                # Now add the constraint -Ry_abs <= t[i] <= Ry_abs
                milp.addLConstr([torch.tensor([-1, -1], dtype=dtype)],
                                [[t[i], Ry_abs[0]]],
                                sense=gurobipy.GRB.LESS_EQUAL,
                                rhs=0.)
                milp.addLConstr([torch.tensor([1, -1], dtype=dtype)],
                                [[t[i], Ry_abs[0]]],
                                sense=gurobipy.GRB.LESS_EQUAL,
                                rhs=0.)
                ti_lo = -torch.maximum(-Ry_lo[i], Ry_up[i])
                ti_up = torch.maximum(-Ry_lo[i], Ry_up[i])
            # Now add the constraint such that
            # alpha1_times_t[i] = α[i][1] * t[i]
            Ain_t, Ain_slack, Ain_alpha, rhs = \
                utils.replace_binary_continuous_product(
                    ti_lo, ti_up, dtype=dtype)
            milp.addMConstrs([
                Ain_t.reshape((-1, 1)),
                Ain_slack.reshape((-1, 1)),
                Ain_alpha.reshape((-1, 1))
            ], [[t[i]], [alpha1_times_t[i]], [l1_binary[i][1]]],
                             sense=gurobipy.GRB.LESS_EQUAL,
                             b=rhs)
        # Now add the constraint
        # λ*∂|R(x−x*)|₁/∂x * y = -λα[:][0]ᵀ*R*y + λα[:][2]ᵀ*R*y + λα[:][1]ᵀ*t
        milp.addLConstr([
            -V_lambda * torch.ones(R.shape[0], dtype=dtype),
            V_lambda * torch.ones(R.shape[0], dtype=dtype),
            V_lambda * torch.ones(R.shape[0], dtype=dtype),
            torch.tensor([-1], dtype=dtype)
        ], [
            alpha0_times_Ry, alpha2_times_Ry, alpha1_times_t, dl1dx_times_y_var
        ],
                        sense=gurobipy.GRB.EQUAL,
                        rhs=0.)

    return dl1dx_times_y_var[0]


def _compute_dl1dx_times_y_sampled_subgradient(
        milp: gurobi_torch_mip.GurobiTorchMIP, l1_binary: list, y: list,
        subgradient_binary: list, subgradient_samples: np.array,
        R: torch.Tensor, Ry_lo: torch.Tensor, Ry_up: torch.Tensor,
        V_lambda: float, return_var_name: str) -> gurobipy.Var:
    """
    Adds the constraint z = λ*∂|R(x−x*)|₁/∂x * y
    The value of z is unique when the gradient ∂|R(x−x*)|₁/∂x is unique.
    When ∂|R(x−x*)|₁/∂x isn't unique, then we consider only a finite number
    of sampled subgradients.

    We write
    ∂|R(x−x*)|₁/∂x * y = -α[:][0]ᵀ*R*y + α[:][2]ᵀ*R*y + ∑ⱼβ[:][j]*s[j]* R * y

    α is the binary variable with meaning
    α[i][0] = 1 => (R*(x-x*))[i] <= 0
    α[i][1] = 1 => (R*(x-x*))[i] = 0
    α[i][2] = 1 => (R*(x-x*))[i] >= 0

    s[j] is the j'th sampled subgradient. β[i][j] is the binary variable
    indicating that the j'th subgradient is active for R.row(i) * (x - x*).

    The user should impose the following constraint separately outside of this
    function:
    ∑ⱼ β[i][j] <= α[i][1] for each i.
    Namely the at most one subgradient can be active when R.row(i) * (x-x*) = 0

    Args:
      l1_binary: α in the documentation above.
      subgradient_binary: β in the documentation above.
      subgradient_samples: A array of number within (-1, 1)
    Return:
      dl1dx_times_y_var: The variable z above, z = λ*∂|R(x−x*)|₁/∂x * y
    """
    assert (isinstance(subgradient_binary, list))
    assert (len(subgradient_binary) == R.shape[0])
    assert (isinstance(subgradient_samples, np.ndarray))
    assert (np.all(subgradient_samples > -1))
    assert (np.all(subgradient_samples < 1))
    assert (all(
        [len(v) == subgradient_samples.size for v in subgradient_binary]))
    assert (all([len(v) == 3 for v in l1_binary]))
    dtype = R.dtype
    alpha0_times_Ry, alpha2_times_Ry = _compute_alpha_times_Ry(
        milp, y, l1_binary, R, Ry_lo, Ry_up)
    # Now add the term beta_times_Ry[i][j] = β[i][j] * R[i, :] * y
    beta_times_Ry = [None] * R.shape[0]
    dl1dx_times_y_var = milp.addVars(1,
                                     lb=-gurobipy.GRB.INFINITY,
                                     name=return_var_name)
    # We will need to add the constraint
    # λ∂|R(x−x*)|₁/∂x * y = -λ*α[:][0]ᵀ*R*y + λ*α[:][2]ᵀ*R*y +
    #                       λ*∑ⱼβ[:][j]*s[j]* R * y
    # The right-hand side is written as
    # dl1dx_times_y_vars * dl1dx_times_y_coeffs.
    dl1dx_times_y_coeffs = [
        -V_lambda * torch.ones(R.shape[0], dtype=dtype),
        V_lambda * torch.ones(R.shape[0], dtype=dtype)
    ]
    dl1dx_times_y_vars = [alpha0_times_Ry, alpha2_times_Ry]
    for i in range(R.shape[0]):
        beta_times_Ry[i] = milp.addVars(subgradient_samples.size,
                                        lb=-gurobipy.GRB.INFINITY)
        A_Ry, A_slack, A_beta, rhs = utils.replace_binary_continuous_product(
            Ry_lo[i], Ry_up[i], dtype=dtype)
        A_y = A_Ry.reshape((-1, 1)) @ R[i, :].reshape((1, -1))
        for j in range(subgradient_samples.size):
            milp.addMConstrs(
                [A_y, A_slack.reshape((-1, 1)),
                 A_beta.reshape((-1, 1))],
                [y, [beta_times_Ry[i][j]], [subgradient_binary[i][j]]],
                sense=gurobipy.GRB.LESS_EQUAL,
                b=rhs)
        dl1dx_times_y_coeffs.append(V_lambda *
                                    torch.from_numpy(subgradient_samples))
        dl1dx_times_y_vars.append(beta_times_Ry[i])
    # Add the constraint
    # λ∂|R(x−x*)|₁/∂x * y =
    #    -λ*α[:][0]ᵀ*R*y + λ*α[:][2]ᵀ*R*y + λ*∑ⱼβ[:][j]*s[j]* R * y
    dl1dx_times_y_coeffs.append(torch.tensor([-1], dtype=dtype))
    dl1dx_times_y_vars.append(dl1dx_times_y_var)
    milp.addLConstr(dl1dx_times_y_coeffs,
                    dl1dx_times_y_vars,
                    sense=gurobipy.GRB.EQUAL,
                    rhs=0.)
    return dl1dx_times_y_var[0]


def _compute_alpha_times_Ry(milp: gurobi_torch_mip.GurobiTorchMIP, y: list,
                            l1_binary: list, R: torch.Tensor,
                            Ry_lo: torch.Tensor, Ry_up: torch.Tensor):
    """
    Compute
    alpha0_times_Ry[i] = α[i][0] * R.row(i) * y
    alpha2_times_Ry[i] = α[i][2] * R.row(i) * y
    where α is the binary variable with meaning
    α[i][0] = 1 => (R*(x-x*))[i] <= 0
    α[i][1] = 1 => (R*(x-x*))[i] = 0
    α[i][2] = 1 => (R*(x-x*))[i] >= 0
    """
    dtype = R.dtype
    # alpha0_times_Ry[i] = α[i][0] * R.row(i) * y
    alpha0_times_Ry = milp.addVars(R.shape[0], lb=-gurobipy.GRB.INFINITY)
    # alpha2_times_Ry[i] = α[i][2] * R.row(i) * y
    alpha2_times_Ry = milp.addVars(R.shape[0], lb=-gurobipy.GRB.INFINITY)
    for i in range(R.shape[0]):
        A_Ry, A_slack, A_alpha, rhs = utils.replace_binary_continuous_product(
            Ry_lo[i], Ry_up[i], dtype=dtype)
        A_y = A_Ry.reshape((-1, 1)) @ R[i, :].reshape((1, -1))
        milp.addMConstrs(
            [A_y, A_slack.reshape((-1, 1)),
             A_alpha.reshape(
                 (-1, 1))], [y, [alpha0_times_Ry[i]], [l1_binary[i][0]]],
            sense=gurobipy.GRB.LESS_EQUAL,
            b=rhs)
        milp.addMConstrs(
            [A_y, A_slack.reshape((-1, 1)),
             A_alpha.reshape(
                 (-1, 1))], [y, [alpha2_times_Ry[i]], [l1_binary[i][2]]],
            sense=gurobipy.GRB.LESS_EQUAL,
            b=rhs)
    return alpha0_times_Ry, alpha2_times_Ry
