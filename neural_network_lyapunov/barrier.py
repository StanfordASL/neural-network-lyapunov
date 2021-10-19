import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.mip_utils as mip_utils

import gurobipy
import torch


class Barrier:
    """
    This is a super class for DiscreteTimeBarrier, ContinuousTimeBarrier and
    ControlBarrier.

    Given an unsafe set Cᵤ, and a known safe state x*, we design a barrier
    function h(x) satisfying
    h(x) < 0 ∀ x ∈ Cᵤ
    ḣ(x) > -ε h(x)
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
        self.system = system
        self.barrier_relu = barrier_relu
        self.barrier_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            barrier_relu, self.system.dtype)
        self.network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.IA

    def barrier_value(self, x, x_star, c: float):
        """
        Compute the value of ϕ(x) - ϕ(x*) + c
        """
        assert (isinstance(c, float))
        assert (c > 0)
        return self.barrier_relu(x) - self.barrier_relu(x_star) + c

    def barrier_unsafe_as_milp(
            self, x_star, c: float,
            unsafe_region_cnstr: gurobi_torch_mip.MixedIntegerConstraintsReturn
    ):
        """
        To compute the maximal violation of the constraint that h(x) is
        negative in the unsafe region, formulate the following optimization
        problem
        max h(x)
        s.t x ∈ Cᵤ
        where Cᵤ is the unsafe region.

        Args:
          x_star: x* in the class documentation
          c: h(x) = ϕ(x) - ϕ(x*) + c
          unsafe_region_cnstr: The mixed-integer constraint that describes the
          unsafe region Cᵤ.

        Return:
          milp: The gurobi_torch_mip.GurobiTorchMILP object that captures the
          maximization problem above.
          x: The state variable.
        """
        assert (isinstance(unsafe_region_cnstr,
                           gurobi_torch_mip.MixedIntegerConstraintsReturn))
        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)
        x = milp.addVars(self.system.x_dim,
                         lb=-gurobipy.GRB.INFINITY,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="x")
        unsafe_slack, unsafe_binary = \
            milp.add_mixed_integer_linear_constraints(
                unsafe_region_cnstr,
                x,
                None,
                "unsafe_s",
                "unsafe_binary",
                "unsafe_ineq",
                "unsafe_eq",
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
        milp.setObjective([barrier_mip_cnstr_return.Aout_slack.squeeze(0)],
                          [barrier_relu_slack],
                          barrier_mip_cnstr_return.Cout.squeeze() -
                          self.barrier_relu(x_star).squeeze() + c,
                          sense=gurobipy.GRB.MAXIMIZE)
        return milp, x
