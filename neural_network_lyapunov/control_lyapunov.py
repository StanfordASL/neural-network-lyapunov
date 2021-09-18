import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.mip_utils as mip_utils
import torch
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
    def __init__(self, milp, x, relu_beta, l1_binary, dl1dx_times_f_slack,
                 system_constraint_return):
        """
        Args:
          relu_beta: The binary variable indicating the activeness of each ReLU
          unit in the network.
          dl1dx_times_f_slack: refer to _add_dl1dx_times_f() for more details.
        """
        self.milp = milp
        self.x = x
        self.beta = relu_beta
        self.l1_binary = l1_binary
        self.dl1dx_times_f_slack = dl1dx_times_f_slack
        self.system_constraint_return = system_constraint_return


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
                 lyapunov_relu):
        """
        Args:
          system: A control-affine system.
          lyapunov_relu: The neural network ϕ which defines the Lyapunov
          function.
        """
        assert (isinstance(system,
                           control_affine_system.ControlPiecewiseAffineSystem))
        super(ControlLyapunov, self).__init__(system, lyapunov_relu)

    def lyapunov_derivative(self,
                            x,
                            x_equilibrium,
                            V_lambda,
                            epsilon,
                            *,
                            R,
                            subgradient_rule: str = "max",
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
          subgradient_rule: Can be either "max", "min", or "all".
            If "max", then we compute the maximal of V̇ among all possible
            subgradients.
            If "min", then we compute the minimal of V̇ among all possible
            subgradients.
            If "all", then we compute all V̇ for every possible subgradients.
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
        dl1_dx = V_lambda * utils.l1_gradient(R @ (x - x_equilibrium),
                                              zero_tol=zero_tol) @ R

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
        if subgradient_rule == "max":
            return torch.max(Vdot.squeeze()) + epsilon * V
        elif subgradient_rule == "min":
            return torch.min(Vdot.squeeze()) + epsilon * V
        elif subgradient_rule == "all":
            return Vdot.squeeze() + epsilon * V

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
        assert (len(f) == self.system.x_dim)
        assert (len(Gt) == self.system.u_dim)
        # Add constraint that x_lo <= x <= x_up
        milp.addMConstrs(
            [torch.eye(self.system.x_dim, dtype=self.system.dtype)], [x],
            gurobipy.GRB.LESS_EQUAL,
            self.system.x_up,
            name="x_up")
        milp.addMConstrs(
            [torch.eye(self.system.x_dim, dtype=self.system.dtype)], [x],
            gurobipy.GRB.GREATER_EQUAL,
            self.system.x_lo,
            name="x_lo")
        # Set the bounds of x
        for i in range(self.system.x_dim):
            if x[i].lb < self.system.x_lo[i].item():
                x[i].lb = self.system.x_lo[i].item()
            if x[i].ub > self.system.x_up[i].item():
                x[i].ub = self.system.x_up[i].item()
        mip_cnstr_ret = self.system.mixed_integer_constraints()
        slack, binary = milp.add_mixed_integer_linear_constraints(
            mip_cnstr_ret.mip_cnstr_f, x, f, "slack_f", "binary_f", "f_ineq",
            "f_eq", "f_output", binary_var_type)
        G_flat = [None] * self.system.x_dim * self.system.u_dim
        for i in range(self.system.x_dim):
            for j in range(self.system.u_dim):
                G_flat[i * self.system.u_dim + j] = Gt[j][i]
        slack_G, binary_G = milp.add_mixed_integer_linear_constraints(
            mip_cnstr_ret.mip_cnstr_G, x, G_flat, "slack_G", "binary_G",
            "G_ineq", "G_eq", "G_out", binary_var_type)
        ret = ControlAffineSystemAddConstraintReturn(slack, binary)
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
            binary_var_type=binary_var_type)
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
        dl1dx_times_f_slack = self._add_dl1dx_times_f(milp, x, l1_binary, f, R,
                                                      Rf_lo, Rf_up, V_lambda,
                                                      Vdot_coeff, Vdot_vars)
        dVdx_times_G_ret, dVdx_times_G_binary = self._add_dVdx_times_G(
            milp, x, l1_binary, relu_beta, Gt, R,
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
                                   dl1dx_times_f_slack,
                                   system_constraint_return)

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
            subgradient_rule = "max"
        elif eps_type == lyapunov.ConvergenceEps.ExpUpper:
            subgradient_rule = "min"

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
                           x: list, l1_binary: list, f: list, R: torch.Tensor,
                           Rf_lo: torch.Tensor, Rf_up: torch.Tensor,
                           V_lambda: float, Vdot_coeff: list,
                           Vdot_vars: list) -> list:
        """
        Adds λ*∂|R(x−x*)|₁/∂x * f to V̇.
        We need to add the mixed-integer linear constraints, and also append
        new terms to Vdot_coeff and Vdot_vars.
        As we have introduced binary variables α for the l1 norm, such that
        α = 1 => (R(x-x*))(i) >= 0
        α = 0 => (R(x-x*))(i) <= 0
        we know that ∂|R(x−x*)|₁/∂x = (2α-1)ᵀ * R. Hence
        λ*∂|R(x−x*)|₁/∂x * f = 2λ*αᵀ*R*f − λ*1ᵀ*R*f

        Args:
          l1_binary: α in the documentation above.
          f: The variables representing the dynamics term f
          Rf_lo, Rf_up: The lower/upper bounds of R*f

        Return:
          slack: the newly added slack variable, slack(i) =  α(i)*(R*f)(i)
        """
        # We need to add slack variables to represent the product between the
        # binary variable α and R*f
        assert (len(l1_binary) == R.shape[0])
        slack = milp.addVars(len(l1_binary),
                             lb=-gurobipy.GRB.INFINITY,
                             name="l1_binary_times_Rf")
        for i in range(len(l1_binary)):
            A_Rf, A_slack, A_alpha, rhs = \
                utils.replace_binary_continuous_product(
                    Rf_lo[i], Rf_up[i], dtype=self.system.dtype)
            A_f = A_Rf.reshape((-1, 1)) @ R[i].reshape((1, -1))
            milp.addMConstrs(
                [A_f, A_slack.reshape((-1, 1)),
                 A_alpha.reshape((-1, 1))], [f, [slack[i]], [l1_binary[i]]],
                sense=gurobipy.GRB.LESS_EQUAL,
                b=rhs)
        Vdot_coeff.append(2 * V_lambda *
                          torch.ones(len(l1_binary), dtype=self.system.dtype))
        Vdot_vars.append(slack)

        Vdot_coeff.append(-V_lambda *
                          torch.ones(R.shape[0], dtype=self.system.dtype) @ R)
        Vdot_vars.append(f)
        return slack

    def _add_dVdx_times_G(self, milp: gurobi_torch_mip.GurobiTorchMIP, x: list,
                          l1_binary: list, relu_beta: list, Gt: list,
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
            milp, x, relu_beta, l1_binary, Gt, G_flat_lo, G_flat_up, RG_lo,
            RG_up, R, V_lambda)

        u_mid = (self.system.u_lo + self.system.u_up) / 2
        # Add ∂V/∂x*G(x) * (u_lo + u_up)/2 to Vdot
        Vdot_coeff.append(u_mid.reshape((-1, )))
        Vdot_vars.append(compute_dVdx_times_G_return.dVdx_times_G)

        dVdx_times_G_binary = milp.addVars(self.system.u_dim,
                                           vtype=gurobipy.GRB.BINARY,
                                           name="dVdx_times_G_binary")
        for i in range(self.system.u_dim):
            # We add slack and binary variables for the absolute value
            # |∂V/∂x * G.col(i)|
            # We add -|∂V/∂x * G.col(i)| * (u_up(i) - u_lo(i))/2
            if compute_dVdx_times_G_return.dVdx_times_G_lo[i] >= 0:
                # The binary variable equals to 1.
                milp.addLConstr([torch.tensor([1], dtype=self.dtype)],
                                [dVdx_times_G_binary[i]],
                                sense=gurobipy.GRB.EQUAL,
                                rhs=1.)
                dVdx_times_G_binary[i].lb = 1.
                dVdx_times_G_binary[i].ub = 1.
                Vdot_coeff.append(
                    torch.tensor(
                        [-(self.system.u_up[i] - self.system.u_lo[i]) / 2],
                        dtype=self.system.dtype))
                Vdot_vars.append([compute_dVdx_times_G_return.dVdx_times_G[i]])
            elif compute_dVdx_times_G_return.dVdx_times_G_up[i] <= 0:
                # The binary variable equals to 0.
                milp.addLConstr([torch.tensor([1], dtype=self.dtype)],
                                [dVdx_times_G_binary[i]],
                                sense=gurobipy.GRB.EQUAL,
                                rhs=0.)
                dVdx_times_G_binary[i].lb = 0.
                dVdx_times_G_binary[i].ub = 0.
                Vdot_coeff.append(
                    torch.tensor(
                        [(self.system.u_up[i] - self.system.u_lo[i]) / 2],
                        dtype=self.system.dtype))
                Vdot_vars.append([compute_dVdx_times_G_return.dVdx_times_G[i]])
            else:
                Ain_abs_input, Ain_abs_output, Ain_abs_binary, rhs_in_abs = \
                    utils.replace_absolute_value_with_mixed_integer_constraint(
                        compute_dVdx_times_G_return.dVdx_times_G_lo[i],
                        compute_dVdx_times_G_return.dVdx_times_G_up[i],
                        dtype=self.system.dtype)
                slack = milp.addVars(1,
                                     lb=-gurobipy.GRB.INFINITY,
                                     name=f"dVdx_times_G[{i}]_abs")
                milp.addMConstrs([
                    Ain_abs_input.reshape((-1, 1)),
                    Ain_abs_output.reshape((-1, 1)),
                    Ain_abs_binary.reshape((-1, 1))
                ], [[compute_dVdx_times_G_return.dVdx_times_G[i]], slack,
                    [dVdx_times_G_binary[i]]],
                                 sense=gurobipy.GRB.LESS_EQUAL,
                                 b=rhs_in_abs,
                                 name=f"dVdx_times_G{i}_abs")
                Vdot_coeff.append(
                    torch.tensor(
                        [-(self.system.u_up[i] - self.system.u_lo[i]) / 2],
                        dtype=self.system.dtype))
                Vdot_vars.append(slack)

        return compute_dVdx_times_G_return, dVdx_times_G_binary

    def validate_x_equilibrium(self, x_equilibrium: torch.Tensor):
        super(ControlLyapunov, self).validate_x_equilibrium(x_equilibrium)
        assert (self.system.can_be_equilibrium_state(x_equilibrium))

    class ComputeDvdxTimesGReturn:
        def __init__(self, dVdx_times_G, dVdx_times_G_lo, dVdx_times_G_up,
                     dphidx_times_G, l1_binary_times_RG):
            """
            dphidx_times_G: ∂ϕ/∂x*G(x)
            l1_binary_times_RG: the newly added slack variable,
            l1_binary_times_RG[i][j] = α(j)*(R*G.col(i))(j)
            """
            self.dVdx_times_G = dVdx_times_G
            self.dVdx_times_G_lo = dVdx_times_G_lo
            self.dVdx_times_G_up = dVdx_times_G_up
            self.dphidx_times_G = dphidx_times_G
            self.l1_binary_times_RG = l1_binary_times_RG

    def _compute_dVdx_times_G(self, milp: gurobi_torch_mip.GurobiTorchMIP,
                              x: list, relu_beta: list, l1_binary: list,
                              Gt: list, G_flat_lo: torch.Tensor,
                              G_flat_up: torch.Tensor, RG_lo: torch.Tensor,
                              RG_up: torch.Tensor, R: torch.Tensor,
                              V_lambda: float):
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

        # l1_binary_times_RG[i][j] = α(j)*(R*G.col(i))(j)
        l1_binary_times_RG = [None] * self.system.u_dim
        for i in range(self.system.u_dim):
            l1_binary_times_RG[i] = milp.addVars(len(l1_binary),
                                                 lb=-gurobipy.GRB.INFINITY,
                                                 name=f"l1_binary_times_RG{i}")
            for j in range(len(l1_binary)):
                A_RG, A_slack, A_alpha, rhs = \
                    utils.replace_binary_continuous_product(
                        RG_lo[j][i], RG_up[j][i], dtype=dtype)
                # A_RG is the coefficient of R.row(j) * G.col(i). Hence the
                # coefficient for G.col(i) is A_RG * R.row(j)
                A_Gi = A_RG.reshape((-1, 1)) @ R[j].reshape((1, -1))
                milp.addMConstrs(
                    [A_Gi,
                     A_slack.reshape((-1, 1)),
                     A_alpha.reshape((-1, 1))],
                    [Gt[i], [l1_binary_times_RG[i][j]], [l1_binary[j]]],
                    sense=gurobipy.GRB.LESS_EQUAL,
                    b=rhs)

        # ∂V/∂x * G(x) = ∂ϕ/∂x*G(x) + λ*∂|R(x−x*)|₁/∂x*G(x)
        # = ∂ϕ/∂x*G(x) + 2λ*αᵀ*R*G − λ*1ᵀ*R*G
        # Hence we add the constraint
        # ∂ϕ/∂x*G + 2λ*αᵀ*R*G - ∂V/∂x * G - λ*1ᵀ*R*G(x) = 0
        for i in range(self.system.u_dim):
            milp.addLConstr([
                torch.tensor([1], dtype=dtype),
                2 * V_lambda * torch.ones(len(l1_binary), dtype=dtype),
                -torch.tensor([1], dtype=dtype),
                -V_lambda * torch.ones(len(l1_binary), dtype=dtype) @ R
            ], [[dphidx_times_G[i]], l1_binary_times_RG[i], [dVdx_times_G[i]],
                Gt[i]],
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
                                                       l1_binary_times_RG)
