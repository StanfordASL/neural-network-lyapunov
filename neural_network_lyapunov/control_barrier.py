import neural_network_lyapunov.barrier as barrier
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.mip_utils as mip_utils

import torch
import gurobipy


class BarrierDerivMilpReturn:
    def __init__(self, milp, x, barrier_relu_binary, system_mip_cnstr_ret,
                 binary_f, binary_G, dhdx_times_G_binary):
        """
        Args:
          barrier_relu_binary: Binary variables in computing the ReLU network
          ϕ(x) in the barrier function.
          binary_f: Binary variable in computing f(x) in the system dynamics.
          binary_G: Binary variable in computing G(x) in the system dynamics
          dhdx_times_G_binary: Binary variable in the absolute value
          |∂h/∂x * G(x)|
        """
        self.milp = milp
        self.x = x
        self.barrier_relu_binary = barrier_relu_binary
        self.system_mip_cnstr_ret = system_mip_cnstr_ret
        self.binary_f = binary_f
        self.binary_G = binary_G
        self.dhdx_times_G_binary = dhdx_times_G_binary


class ControlBarrier(barrier.Barrier):
    """
    For a continuous-time control affine system ẋ = f(x) + G(x)u whose input u
    is bounded u_lo <= u <= u_up, we aim to find a control barrier function
    h(x) to guarantee the safety of the super-level set {x | h(x) >= 0}.

    Note that barrier function requires
    maxᵤ ∂h/∂x*f(x) + ∂h/∂x*G(x)u ≥ −ε h(x)
    s.t u_lo <= u <= u_up
    And this maximal over u has an analytical form as
    maxᵤ ∂h/∂x*f(x) + ∂h/∂x*G(x)u
    s.t u_lo <= u <= u_up
    is equivalent to
    ∂h/∂x*f(x)+∂h/∂x*G(x)*(u_lo+u_up)/2 + |∂h/∂x*G(x)*diag(u_up-u_lo)/2|₁
    """
    def __init__(self, system, barrier_relu):
        assert (isinstance(system,
                           control_affine_system.ControlPiecewiseAffineSystem))
        super(ControlBarrier, self).__init__(system, barrier_relu)

    def barrier_derivative(self, x: torch.Tensor, *, zero_tol=0.):
        """
        Evaluates maxᵤ ∂h/∂x*f(x) + ∂h/∂x*G(x)u
                  s.t u_lo <= u <= u_up
        If ∂h/∂x is not unique (some ReLU unit has input 0), then we return all
        subgradients with both the left and right derivatives.
        """
        assert (x.shape == (self.system.x_dim, ))
        barrier_grad = utils.relu_network_gradient(
            self.barrier_relu, x, zero_tol=zero_tol).squeeze(1)
        u_mid = (self.system.u_lo + self.system.u_up) / 2
        delta_u = (self.system.u_up - self.system.u_lo) / 2
        f = self.system.f(x)
        G = self.system.G(x)
        hdot = barrier_grad @ (f + G @ u_mid) + torch.norm(
            (barrier_grad @ G) * delta_u.repeat((barrier_grad.shape[0], 1)),
            p=1,
            dim=1)
        return hdot

    def barrier_derivative_as_milp(self,
                                   x_star,
                                   c: float,
                                   epsilon: float,
                                   *,
                                   binary_var_type=gurobipy.GRB.BINARY):
        """
        Formulate the program
        maxₓ −ḣ(x)−εh(x)
             x∈ℬ
        as a mixed-integer program.
        If the optimal value is non-positive, then we prove
        ḣ(x) ≥ -ε h(x) ∀x∈ℬ
        ℬ is a bounded set (normally x_lo <= x <= x_up).

        Args:
          x_star, c: Our barrier function is defined as h(x) = ϕ(x) − ϕ(x*) + c
          epsilon: a positive constant.
        """
        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)
        x = milp.addVars(self.system.x_dim,
                         lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="x")
        f = milp.addVars(self.system.x_dim,
                         lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="f")
        # Gt[i] is system.G(x).col(i)
        Gt = [None] * self.system.u_dim
        for i in range(self.system.u_dim):
            Gt[i] = milp.addVars(self.system.x_dim,
                                 lb=-gurobipy.GRB.INFINITY,
                                 vtype=gurobipy.GRB.CONTINUOUS,
                                 name=f"Gt[{i}]")
        system_mip_cnstr_ret, _, _, binary_f, binary_G = \
            control_affine_system.add_system_constraint(
                self.system, milp, x, f, Gt, binary_var_type=binary_var_type)

        # We need to compute ∂ϕ/∂x*f(x) and ∂ϕ/∂x*G(x), which requires the
        # binary variables indicating the activeness of each ReLU unit.
        barrier_mip_cnstr_return = \
            self.barrier_relu_free_pattern.output_constraint(
                torch.from_numpy(self.system.x_lo_all),
                torch.from_numpy(self.system.x_up_all),
                self.network_bound_propagate_method)
        barrier_relu_slack, barrier_relu_binary = \
            milp.add_mixed_integer_linear_constraints(
                barrier_mip_cnstr_return,
                x,
                None,
                "barrier_relu_s",
                "barrier_relu_binary",
                "barrier_relu_ineq",
                "barrier_relu_eq",
                "",
                binary_var_type=binary_var_type)

        mip_cnstr_dphidx_times_f = \
            self.barrier_relu_free_pattern.output_gradient_times_vector(
                system_mip_cnstr_ret.f_lo, system_mip_cnstr_ret.f_up)

        dphidx_times_f_slack, _ = milp.add_mixed_integer_linear_constraints(
            mip_cnstr_dphidx_times_f, f, None, "f_slack", barrier_relu_binary,
            "f_ineq", "f_eq", "", binary_var_type)
        # I expect these entries to be None, so that cost_coeff/cost_vars won't
        # include these entries.
        assert (mip_cnstr_dphidx_times_f.Aout_input is None)
        assert (mip_cnstr_dphidx_times_f.Aout_binary is None)
        assert (mip_cnstr_dphidx_times_f.Cout is None)
        # cost will be cost_coeff * cost_vars + cost_constant.
        cost_coeff = [-mip_cnstr_dphidx_times_f.Aout_slack.reshape((-1, ))]
        cost_vars = [dphidx_times_f_slack]
        cost_constant = torch.tensor(0, dtype=self.system.dtype)
        # Add -max_u ∂h/∂x * G(x) * u s.t u_lo <= u <= u_up to the cost.
        dhdx_times_G, dhdx_times_G_binary = self._add_dhdx_times_G(
            milp, x, barrier_relu_binary, Gt, system_mip_cnstr_ret.G_flat_lo,
            system_mip_cnstr_ret.G_flat_up, cost_coeff, cost_vars,
            binary_var_type)
        # Add the term -εh(x)=−εϕ(x) + εϕ(x*)−εc to the cost
        cost_coeff.append(-epsilon *
                          barrier_mip_cnstr_return.Aout_slack.squeeze(0))
        cost_vars.append(barrier_relu_slack)
        cost_constant += -epsilon * barrier_mip_cnstr_return.Cout.squeeze(
            0) + epsilon * self.barrier_relu(x_star).squeeze() - epsilon * c

        milp.setObjective(cost_coeff,
                          cost_vars,
                          cost_constant,
                          sense=gurobipy.GRB.MAXIMIZE)
        return BarrierDerivMilpReturn(milp, x, barrier_relu_binary,
                                      system_mip_cnstr_ret, binary_f, binary_G,
                                      dhdx_times_G_binary)

    def _add_dhdx_times_G(self, milp: gurobi_torch_mip.GurobiTorchMIP, x: list,
                          barrier_relu_binary: list, Gt: list,
                          G_flat_lo: torch.Tensor, G_flat_up: torch.Tensor,
                          cost_coeff: list, cost_vars: list, binary_var_type):
        """
        Add -max_u ∂h/∂x * G * u
            s.t  u_lo <= u <= u_up
        to cost
        """
        dhdx_times_G, dhdx_times_G_lo, dhdx_times_G_up = \
            self._compute_dhdx_times_G(
                milp, x, barrier_relu_binary, Gt, G_flat_lo, G_flat_up,
                binary_var_type)
        # We know that
        # -max_u ∂h/∂x * G * u
        #    s.t  u_lo <= u <= u_up
        # equals to
        # -(∂h/∂x*G) * (u_lo+u_up)/2 - |(∂h/∂x*G) * diag(u_up-u_lo)/2|₁
        # Now add the terms for - |(∂h/∂x*G) * diag(u_up-u_lo)/2|₁
        dhdx_times_G_binary = milp.addVars(self.system.u_dim,
                                           vtype=gurobipy.GRB.BINARY,
                                           name="dhdx_times_G_binary")
        for i in range(self.system.u_dim):
            if dhdx_times_G_lo[i] >= 0:
                # Add the cost -∂h/∂x*G)[i] * u_up[i]
                # and the constraint dhdx_times_G_binary[i] = 1
                milp.addLConstr([torch.tensor([1], dtype=self.system.dtype)],
                                [[dhdx_times_G_binary[i]]],
                                sense=gurobipy.GRB.EQUAL,
                                rhs=1.)
                dhdx_times_G_binary[i].lb = 1.
                dhdx_times_G_binary[i].ub = 1.
                cost_vars.append([dhdx_times_G[i]])
                cost_coeff.append((-self.system.u_up[i]).reshape((1, )))
            elif dhdx_times_G_up[i] < 0:
                # Add the cost -∂h/∂x*G)[i] * u_lo[i]
                # and the constraint dhdx_times_G_binary[i] = 0
                milp.addLConstr([torch.tensor([1], dtype=self.system.dtype)],
                                [[dhdx_times_G_binary[i]]],
                                sense=gurobipy.GRB.EQUAL,
                                rhs=0.)
                dhdx_times_G_binary[i].lb = 0.
                dhdx_times_G_binary[i].ub = 0.
                cost_vars.append([dhdx_times_G[i]])
                cost_coeff.append((-self.system.u_lo[i]).reshape((1, )))
            else:
                # Need to introduce new slack variable and binary variable for
                # |(∂h/∂x*G)[i]|
                mip_cnstr_abs = \
                    utils.absolute_value_as_mixed_integer_constraint(
                        dhdx_times_G_lo[i],
                        dhdx_times_G_up[i],
                        binary_for_zero_input=False)
                slack, _ = milp.add_mixed_integer_linear_constraints(
                    mip_cnstr_abs, [dhdx_times_G[i]],
                    None,
                    f"dhdx_times_G[{i}]_abs", [dhdx_times_G_binary[i]],
                    "",
                    "",
                    "",
                    binary_var_type=binary_var_type)
                # Add the cost -(∂h/∂x*G)[i] * (u_lo[i]+u_up[i])/2 -
                # |(∂h/∂x*G)[i]| * (u_up[i] - u_lo[i])/2
                cost_vars.append([dhdx_times_G[i]] + slack)
                cost_coeff.append(
                    torch.stack(
                        (-(self.system.u_lo[i] + self.system.u_up[i]) / 2,
                         (-(self.system.u_up[i] - self.system.u_lo[i]) / 2))))
        return dhdx_times_G, dhdx_times_G_binary

    def _compute_dhdx_times_G(self, milp: gurobi_torch_mip.GurobiTorchMIP,
                              x: list, barrier_relu_binary: list, Gt: list,
                              G_flat_lo: torch.Tensor, G_flat_up: torch.Tensor,
                              binary_var_type):
        """
        Add constraint for ∂h/∂x * G(x) to the program.
        We introduce a new variable dhdx_times_G, add the constraint
        dhdx_times_G = ∂h/∂x * G(x).
        """
        dhdx_times_G = milp.addVars(self.system.u_dim,
                                    lb=-gurobipy.GRB.INFINITY,
                                    name="dhdx_times_G")
        # The mip constraints for ∂h/∂x * G(x).col(i)
        mip_cnstr_dhdx_times_G = [None] * self.system.u_dim
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
                mip_cnstr_dhdx_times_G[j] = self.barrier_relu_free_pattern.\
                    output_gradient_times_vector(Gi_lo, Gi_up)
        else:
            raise NotImplementedError
        dhdx_times_G = milp.addVars(self.system.u_dim,
                                    lb=-gurobipy.GRB.INFINITY,
                                    name="dhdx_times_G")
        for i in range(self.system.u_dim):
            milp.add_mixed_integer_linear_constraints(
                mip_cnstr_dhdx_times_G[i], Gt[i], [dhdx_times_G[i]],
                f"dhdx_times_G[{i}]_slack", barrier_relu_binary,
                f"dhdx_times_G[{i}]_ineq", f"dhdx_times_G[{i}]_eq",
                f"dhdx_times_G[{i}]_out", binary_var_type)
        dhdx_times_G_lo = torch.stack([
            mip_cnstr_dhdx_times_G[i].Wz_lo[-1].squeeze()
            for i in range(self.system.u_dim)
        ])
        dhdx_times_G_up = torch.stack([
            mip_cnstr_dhdx_times_G[i].Wz_up[-1].squeeze()
            for i in range(self.system.u_dim)
        ])
        return dhdx_times_G, dhdx_times_G_lo, dhdx_times_G_up
