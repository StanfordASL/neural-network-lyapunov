import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.utils as utils

import gurobipy
import torch


class BarrierValueMilpReturn:
    def __init__(self, milp, x):
        self.milp = milp
        self.x = x
        self.inf_norm_binary = None


class InfNormTerm:
    """
    Add a term -|Rx-p|∞ to the barrier function h(x).
    The intuition is that since we require h(x) to be negative at the boundary
    of the verified region, which is often parameterized as the boundary of an
    axis-aligned bounding box, we can subtrace |Rx-p|∞ to drive h(x) being
    negative at the boundary of the box.
    """
    def __init__(self, R: torch.Tensor = None, p: torch.Tensor = None):
        self.R = R
        self.p = p

    def check(self):
        assert ((self.R is None and self.p is None)
                or (self.R.shape[0] == self.p.shape[0]))


class Barrier:
    """
    This is a super class for DiscreteTimeBarrier, ContinuousTimeBarrier and
    ControlBarrier.

    Given an unsafe set Cᵤ, and a known safe state x*, we design a barrier
    function h(x) satisfying
    h(x) < 0 ∀ x ∈ Cᵤ
    ḣ(x) ≥ -ε h(x)
    h(x*) > 0
    where ε is a positive constant. This barrier function guarantees that the
    super-level set {x | h(x) >= 0} is safe and invariant.

    Since we only verify the barrier function condition on a bounded set of
    state ℬ, to ensure that the super-level set is contained within the set ℬ,
    we also want to verify the condition
    h(x) < 0 ∀ x∈∂ℬ
    where ∂ℬ is the boundary of the set ℬ.

    We design the barrier function as
    h(x) = ϕ(x) − ϕ(x*) + c - |Rx-p|∞
    where c is a given positive constant. ϕ(x) is a neural network with (leaky)
    ReLU activations. This barrier function satisfies h(x*) > 0.
    If either (or both) R, p are None, then we don't add the term -|Rx-p|∞
    """
    def __init__(self, system, barrier_relu):
        """
        Args:
          system: The dynamical system.
          barrier_relu: ϕ(x) in the class documentation.
        """
        self.system = system
        self.barrier_relu = barrier_relu
        self.barrier_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            barrier_relu, self.system.dtype)
        self.network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.IA

    def barrier_value(self,
                      x,
                      x_star,
                      c: float,
                      inf_norm_term: InfNormTerm = None):
        """
        Compute the value of ϕ(x) - ϕ(x*) + c
        """
        assert (isinstance(c, float))
        assert (c > 0)
        val = self.barrier_relu(x) - self.barrier_relu(x_star) + c
        if (inf_norm_term is None):
            return val
        else:
            if (len(x.shape) == 1):
                return val - torch.norm(inf_norm_term.R @ x - inf_norm_term.p,
                                        p=float("inf"))
            else:
                return val - torch.norm(
                    (inf_norm_term.R @ x.T).T - inf_norm_term.p,
                    p=float("inf"),
                    dim=1).unsqueeze(1)

    def barrier_value_as_milp(
            self,
            x_star,
            c: float,
            region: gurobi_torch_mip.MixedIntegerConstraintsReturn,
            region_name="",
            inf_norm_term: InfNormTerm = None) -> BarrierValueMilpReturn:
        """
        To compute the maximal violation of the constraint that h(x) is
        negative in the region, formulate the following optimization
        problem
        max h(x)
        s.t x ∈ C
        where C is the region.

        Args:
          x_star: x* in the class documentation
          c: h(x) = ϕ(x) - ϕ(x*) + c
          region : The mixed-integer constraint that describes the region C.
          inf_norm_term: If it is not None, then the barrier function has the
          additional term -|Rx-p|∞

        Return:
          milp: The gurobi_torch_mip.GurobiTorchMILP object that captures the
          maximization problem above.
          x: The state variable.
        """
        assert (isinstance(region,
                           gurobi_torch_mip.MixedIntegerConstraintsReturn))
        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)
        x = milp.addVars(self.system.x_dim,
                         lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="x")
        region_slack, region_binary = \
            milp.add_mixed_integer_linear_constraints(
                region,
                x,
                None,
                region_name+"region_s",
                region_name+"region_binary",
                region_name+"region_ineq",
                region_name+"region_eq",
                "",
                binary_var_type=gurobipy.GRB.BINARY)
        barrier_mip_cnstr_return = \
            self.barrier_relu_free_pattern.output_constraint(
                torch.from_numpy(self.system.x_lo_all),
                torch.from_numpy(self.system.x_up_all),
                self.network_bound_propagate_method)
        barrier_relu_slack, _ = milp.add_mixed_integer_linear_constraints(
            barrier_mip_cnstr_return,
            x,
            None,
            "barrier_relu_s",
            "barrier_relu_binary",
            "barrier_relu_ineq",
            "barrier_relu_eq",
            "",
            binary_var_type=gurobipy.GRB.BINARY)
        # The cost function is ϕ(x) − ϕ(x*) + c
        # = a_out * slack + b_out - ϕ(x*) + c
        assert (barrier_mip_cnstr_return.Aout_input is None)
        if barrier_mip_cnstr_return.Cout is None:
            barrier_mip_cnstr_return.Cout = torch.tensor(
                [0], dtype=self.system.dtype)
        objective_coeff = [barrier_mip_cnstr_return.Aout_slack.squeeze(0)]
        objective_var = [barrier_relu_slack]
        if inf_norm_term is not None:
            inf_norm, inf_norm_binary = self._add_inf_norm_term(
                milp, x, inf_norm_term)
            objective_coeff.append(torch.tensor([-1], dtype=self.system.dtype))
            objective_var.append(inf_norm)
        milp.setObjective(objective_coeff,
                          objective_var,
                          barrier_mip_cnstr_return.Cout.squeeze() -
                          self.barrier_relu(x_star).squeeze() + c,
                          sense=gurobipy.GRB.MAXIMIZE)
        ret = BarrierValueMilpReturn(milp, x)
        if inf_norm_term is not None:
            ret.inf_norm_binary = inf_norm_binary
        return ret

    def _add_inf_norm_term(self, milp, x, inf_norm_term):
        """
        Add the constraint y = |Rx-p|∞ to the program.

        Return:
          inf_norm: The variable representing |Rx-p|∞
          inf_norm_binary: inf_norm_binary is of length 2 * R.shape[0],
          inf_norm_binary[i] = 1 if the |Rx-p|∞ = R[i]*x-p[i],
          inf_norm_binary[R.shape[0]+i] = 1 if |Rx-p|∞ = -(R[i]*x-p[i])
        """
        Rx_minus_p_lb, Rx_minus_p_ub = mip_utils.compute_range_by_IA(
            inf_norm_term.R, -inf_norm_term.p, self.system.x_lo,
            self.system.x_up)
        # The infinity norm of R*x-p is the maximal of (R*x-p, -(R*x-p))
        inf_norm_mip_cnstr = utils.max_as_mixed_integer_constraint(
            torch.cat((Rx_minus_p_lb, -Rx_minus_p_ub)),
            torch.cat((Rx_minus_p_ub, -Rx_minus_p_lb)))
        inf_norm_mip_cnstr.transform_input(
            torch.cat((inf_norm_term.R, -inf_norm_term.R), dim=0),
            torch.cat((-inf_norm_term.p, inf_norm_term.p)))
        inf_norm, inf_norm_binary = \
            milp.add_mixed_integer_linear_constraints(
                inf_norm_mip_cnstr,
                x,
                None,
                "inf_norm",
                "inf_norm_binary",
                "inf_norm_ineq",
                "inf_norm_eq",
                "",
                binary_var_type=gurobipy.GRB.BINARY)
        return inf_norm, inf_norm_binary

    def _barrier_gradient(self, x, inf_norm_term):
        """
        Compute the gradient ∂h/∂x for the barrier function h(x).
        If there are multiple possible gradient, then we return all the left
        and right subgradient.
        """
        assert (x.shape == (self.system.x_dim, ))
        dphidx = utils.relu_network_gradient(self.barrier_relu, x).squeeze(1)
        if inf_norm_term is not None:
            dinfnorm_dx = utils.l_infinity_gradient(
                inf_norm_term.R @ x - inf_norm_term.p) @ inf_norm_term.R
            dhdx = utils.minikowski_sum(dphidx, -dinfnorm_dx)
        else:
            dhdx = dphidx
        return dhdx

    def _barrier_gradient_batch(self, x, inf_norm_term, create_graph):
        """
        Compute the gradient ∂h/∂x for the barrier function h(x).
        This function assumes x is a batch of state. When there are multiple
        possible subgradients, we take the one returned from pytorch autodiff.
        """
        assert (x.shape[1] == self.system.x_dim)
        x_star = torch.zeros((self.system.x_dim, ), dtype=self.system.dtype)
        c = 100.
        x_requires_grad = x.requires_grad
        x.requires_grad = True
        dhdx = torch.zeros_like(x, dtype=x.dtype)
        h = self.barrier_value(x, x_star, c, inf_norm_term=inf_norm_term)
        for i in range(x.shape[0]):
            grd = torch.zeros_like(h, dtype=x.dtype)
            grd[i] = 1
            dhdx[i, :] = torch.autograd.grad(
                outputs=h,
                inputs=x,
                grad_outputs=grd,
                retain_graph=True,
                create_graph=create_graph)[0][i, :]
        x.requires_grad = x_requires_grad
        return dhdx
