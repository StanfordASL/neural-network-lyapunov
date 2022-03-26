import torch
import gurobipy
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.mip_utils as mip_utils


class Barrier:
    """
    This is a super class for discrete-time and continuous-time barrier
    function.
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
    h(x) = ϕ(x) − ϕ(x*) + c
    where c is a given positive constant. ϕ(x) is a neural network with (leaky)
    ReLU activations. This barrier function satisfies h(x*) > 0.
    """
    def __init__(self, system, barrier_relu):
        """
        Args:
          system: The dynamical system.
          barrier_relu: ϕ(x) in the class documentation.
        """
        assert (barrier_relu[0].in_features == system.x_dim)
        assert (barrier_relu[-1].out_features == 1)
        self.system = system
        self.barrier_relu = barrier_relu
        self.barrier_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            self.barrier_relu, self.system.dtype)
        self.network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.IA

    def barrier_value(self, x, x_star, c: float):
        """
        Compute the value of ϕ(x) - ϕ(x*) + c
        """
        assert (isinstance(c, float))
        assert (c > 0)
        val = self.barrier_relu(x) - self.barrier_relu(x_star) + c
        if len(x.shape) > 1:
            return val.squeeze()
        else:
            return val

    def barrier_value_as_milp(
            self, x_star, c: float,
            region: gurobi_torch_mip.MixedIntegerConstraintsReturn,
            safe_flag: bool):
        """
        To compute the maximal violation of the constraint that h(x) is
        negative in the unsafe region, or positive in the safe region, we
        formulate the following optimization problem
        max h(x)
        s.t x ∈ Cu
        where Cu is the unsafe region.
        or
        max -h(x)
        s.t x ∈ Cs
        where Cs is the safe region.
        Args:
          x_star: x* in the class documentation
          c: h(x) = ϕ(x) - ϕ(x*) + c
          region : The mixed-integer constraint that describes the region C.
          safe_flag: If true, then region is safe, and the MIP objective is
          max -h(x); Otherwise the region is unsafe and the MIP objective is
          max h(x)
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
                         name="x")
        region_name = "safe_" if safe_flag else "unsafe_"
        region_slack, region_binary = \
            milp.add_mixed_integer_linear_constraints(
                region,
                x,
                None,
                region_name + "slack",
                region_name + "binary",
                region_name + "ineq",
                region_name + "eq",
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
        # h(x) = ϕ(x) - ϕ(x*) + c
        objective_coeff = [barrier_mip_cnstr_return.Aout_slack.squeeze(0)]
        objective_var = [barrier_relu_slack]
        objective_constant = barrier_mip_cnstr_return.Cout.squeeze(
        ) - self.barrier_relu(x_star).squeeze() + c
        if safe_flag:
            objective_coeff = [-objective_coeff[0]]
            objective_constant *= -1
        milp.setObjective(objective_coeff,
                          objective_var,
                          objective_constant,
                          sense=gurobipy.GRB.MAXIMIZE)
        return (milp, x)
