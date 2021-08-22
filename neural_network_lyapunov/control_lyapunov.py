import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.mip_utils as mip_utils
import torch
import gurobipy


class ControlAffineSystemConstraintReturn(lyapunov.SystemConstraintReturn):
    """
    The return type of add_system_constraint()
    """
    def __init__(self, slack, binary):
        super(ControlAffineSystemConstraintReturn,
              self).__init__(slack, binary)
        self.mip_cnstr_f = None
        self.mip_cnstr_G = None


class LyapDerivMilpReturn:
    def __init__(self, milp, x, relu_beta, dphidx_times_G_l1_binary,
                 system_constraint_return):
        """
        Args:
          relu_beta: The binary variable indicating the activeness of each ReLU
          unit in the network.
          dphidx_times_G_l1_binary: When computing
          |∂ϕ/∂x*G(x) * (u_up - u_up)/2|₁, we may need to introduce binary
          variables for the l1 norm.
        """
        self.milp = milp
        self.x = x
        self.relu_beta = relu_beta
        self.dphidx_times_G_l1_binary = dphidx_times_G_l1_binary
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
                            subgradient_rule: str = "max"):
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
        """
        assert (isinstance(x, torch.Tensor))
        assert (x.shape == (self.system.x_dim, ))

        R = lyapunov._get_R(R, self.system.x_dim, x_equilibrium.device)

        # First compute ∂ϕ/∂x
        dphi_dx = utils.relu_network_gradient(self.lyapunov_relu, x).squeeze(1)

        # Now compute the gradient of λ|R(x−x*)|₁
        dl1_dx = V_lambda * utils.l1_gradient(R @ (x - x_equilibrium)) @ R

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
    ) -> ControlAffineSystemConstraintReturn:
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
        mip_cnstr_f, mip_cnstr_G = self.system.mixed_integer_constraints()
        slack, binary = milp.add_mixed_integer_linear_constraints(
            mip_cnstr_f, x, f, "slack_f", "binary_f", "f_ineq", "f_eq",
            "f_output", binary_var_type)
        G_flat = [None] * self.system.x_dim * self.system.u_dim
        for i in range(self.system.x_dim):
            for j in range(self.system.u_dim):
                G_flat[i * self.system.u_dim + j] = Gt[j][i]
        slack_G, binary_G = milp.add_mixed_integer_linear_constraints(
            mip_cnstr_G, x, G_flat, "slack_G", "binary_G", "G_ineq", "G_eq",
            "G_out", binary_var_type)
        ret = ControlAffineSystemConstraintReturn(slack, binary)
        ret.mip_cnstr_f = mip_cnstr_f
        ret.mip_cnstr_G = mip_cnstr_G
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

        # The mip constraints for ∂ϕ/∂x*G(x).col(i).
        mip_cnstr_dphidx_times_G = [None] * self.system.u_dim
        if self.network_bound_propagate_method == \
                mip_utils.PropagateBoundsMethod.IA:
            f_lo, f_up = self.system.compute_f_range_ia()
            G_flat_lo, G_flat_up = self.system.compute_G_range_ia()
            mip_cnstr_dphidx_times_f = \
                self.lyapunov_relu_free_pattern.output_gradient_times_vector(
                    f_lo, f_up)
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
        Vdot_vars = [dphidx_times_f_slack]
        Vdot_constant = torch.tensor(0, dtype=self.system.dtype)
        # An alternative formulation is not to introduce the variable
        # dphidx_times_G, but write it as an affine function of the slack
        # variable in mip_cnstr_dphidx_times_G.
        dphidx_times_G = milp.addVars(self.system.u_dim,
                                      lb=-gurobipy.GRB.INFINITY,
                                      name="dphidx_times_G")
        for i in range(self.system.u_dim):
            milp.add_mixed_integer_linear_constraints(
                mip_cnstr_dphidx_times_G[i], Gt[i], [dphidx_times_G[i]],
                f"G[{i}]_slack", relu_beta, f"G[{i}]_ineq", f"G[{i}]_eq",
                f"G[{i}]", binary_var_type)
        u_mid = (self.system.u_lo + self.system.u_up) / 2
        # Add ∂ϕ/∂x*G(x) * (u_lo + u_up)/2 to Vdot
        Vdot_coeff.append(u_mid.reshape((-1, )))
        Vdot_vars.append(dphidx_times_G)

        # Add -|∂ϕ/∂x*G(x) * (u_up - u_up)/2|₁ to Vdot
        dphidx_times_G_l1_binary = [None] * self.system.u_dim
        for i in range(self.system.u_dim):
            if mip_cnstr_dphidx_times_G[i].Wz_lo[-1] >= 0:
                # ∂ϕ/∂x*Gᵢ(x) >= 0. The absolute value is itself.
                Vdot_coeff.append(
                    -(self.system.u_up[i] - self.system.u_lo[i]).reshape(
                        (-1, )) / 2)
                Vdot_vars.append([dphidx_times_G[i]])
            elif mip_cnstr_dphidx_times_G[i].Wz_up[-1] <= 0:
                # ∂ϕ/∂x*Gᵢ(x) <= 0. The absolute value is its negation.
                Vdot_coeff.append(
                    (self.system.u_up[i] - self.system.u_lo[i]).reshape(
                        (-1, )) / 2)
                Vdot_vars.append([dphidx_times_G[i]])
            else:
                Ain_dphidx_times_G, Ain_dphidx_times_G_l1_slack,\
                    Ain_dphidx_times_G_l1_binary, rhs_in_dphidx_times_G_l1 = \
                    utils.replace_absolute_value_with_mixed_integer_constraint(
                        mip_cnstr_dphidx_times_G[i].Wz_lo[-1].squeeze(),
                        mip_cnstr_dphidx_times_G[i].Wz_up[-1].squeeze(),
                        self.system.dtype)
                dphidx_times_G_l1_slack = milp.addVars(
                    1,
                    lb=-gurobipy.GRB.INFINITY,
                    name=f"dphidx_times_G[{i}]_l1_slack")
                dphidx_times_G_l1_binary[i] = milp.addVars(
                    1,
                    vtype=gurobipy.GRB.BINARY,
                    name=f"dphidx_times_G[{i}]_l1_binary")[0]
                milp.addMConstrs([
                    Ain_dphidx_times_G.reshape((-1, 1)),
                    Ain_dphidx_times_G_l1_slack.reshape((-1, 1)),
                    Ain_dphidx_times_G_l1_binary.reshape((-1, 1))
                ], [[dphidx_times_G[i]], dphidx_times_G_l1_slack,
                    [dphidx_times_G_l1_binary[i]]],
                                 sense=gurobipy.GRB.LESS_EQUAL,
                                 b=rhs_in_dphidx_times_G_l1,
                                 name=f"dphidx_times_G[{i}]_l1")
                Vdot_coeff.append(
                    -0.5 * (self.system.u_up[i] - self.system.u_lo[i]).reshape(
                        (-1, )))
                Vdot_vars.append(dphidx_times_G_l1_slack)

        # We need to compute |R(x−x*)|₁
        # l1_slack, l1_binary = self.add_state_error_l1_constraint(
        #    milp,
        #    x_equilibrium,
        #    x,
        #    R=R,
        #    slack_name="l1_slack",
        #    binary_var_name="l1_binary",
        #    binary_var_type=binary_var_type)
        # TODO(hongkai.dai): implement epsilon != 0 case.
        if epsilon != 0:
            raise NotImplementedError

        milp.setObjective(Vdot_coeff,
                          Vdot_vars,
                          Vdot_constant,
                          sense=gurobipy.GRB.MAXIMIZE)

        return LyapDerivMilpReturn(milp, x, relu_beta,
                                   dphidx_times_G_l1_binary,
                                   system_constraint_return)

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
        Vdot_coeff.append(2 * V_lambda * torch.ones(
            (len(l1_binary), ), dtype=self.system.dtype))
        Vdot_vars.append(slack)

        Vdot_coeff.append(-V_lambda *
                          torch.ones(R.shape[0], dtype=self.system.dtype) @ R)
        Vdot_vars.append(f)
        return slack
