import torch
import gurobipy
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.dynamic_system as dynamic_system


class BarrierDerivReturn:
    def __init__(self, milp, x, x_next, binary_current, binary_next,
                 system_constraint_return, barrier_relu_mip_cnstr_ret):
        self.milp = milp
        self.x = x
        self.x_next = x_next
        self.binary_current = binary_current
        self.binary_next = binary_next
        self.system_constraint_return = system_constraint_return
        self.barrier_relu_mip_cnstr_ret = barrier_relu_mip_cnstr_ret


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

    def value(self, x, x_star, c: float):
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

    def value_as_milp(self, x_star, c: float,
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


class DiscreteTimeBarrier(Barrier):
    def __init__(self, system, barrier_relu):
        super(DiscreteTimeBarrier, self).__init__(system, barrier_relu)

    def derivative(self, x, x_star, c, epsilon):
        """
        Compute -h(x[n+1]) + h(x[n]) - ε*h(x[n])
        Note that we want this value to be <= 0
        """
        x_next = self.system.step_forward(x)
        return -self.value(x_next, x_star, c) + (1 - epsilon) * self.value(
            x, x_star, c)

    def derivative_as_milp(self,
                           x_star,
                           c,
                           epsilon,
                           *,
                           binary_var_type=gurobipy.GRB.BINARY):
        """
        Compute max -h(x[n+1]) + h(x[n]) - ε*h(x[n]) as an MILP.
        The objective is
        −ϕ(x[n+1]) + (1−ε)ϕ(x[n]) + εϕ(x*) - εc
        """
        milp = gurobi_torch_mip.GurobiTorchMILP(self.system.dtype)
        x = milp.addVars(self.system.x_dim,
                         lb=-gurobipy.GRB.INFINITY,
                         name="x")
        x_next = milp.addVars(self.system.x_dim,
                              lb=-gurobipy.GRB.INFINITY,
                              name="x")
        # Add the constraint to compute x_next = f(x)
        system_constraint_return = dynamic_system._add_system_constraint(
            self.system, milp, x, x_next, binary_var_type=binary_var_type)
        # Add the mixed-integer constraint that formulate the output ϕ(x)
        relu_mip_cnstr_return = \
            self.barrier_relu_free_pattern.output_constraint(
                torch.from_numpy(self.system.x_lo_all),
                torch.from_numpy(self.system.x_up_all),
                self.network_bound_propagate_method)
        relu_slack_current, relu_binary_current = \
            milp.add_mixed_integer_linear_constraints(
                relu_mip_cnstr_return, x, None, "relu_slack", "relu_binary",
                "relu_ineq", "relu_eq", "", binary_var_type)
        relu_slack_next, relu_binary_next = \
            milp.add_mixed_integer_linear_constraints(
                relu_mip_cnstr_return, x_next, None, "relu_slack_next",
                "relu_binary_next", "relu_ineq", "relu_eq", "",
                binary_var_type)
        # phi(x) = Aout_slack * slack + Cout
        assert (relu_mip_cnstr_return.Aout_input is None)
        assert (relu_mip_cnstr_return.Aout_binary is None)
        objective_coeff = [
            (1 - epsilon) * relu_mip_cnstr_return.Aout_slack.squeeze(0),
            -relu_mip_cnstr_return.Aout_slack.squeeze(0)
        ]
        objective_vars = [relu_slack_current, relu_slack_next]
        objective_constant = epsilon * (-relu_mip_cnstr_return.Cout.squeeze() +
                                        self.barrier_relu(x_star).squeeze() -
                                        c)
        milp.setObjective(objective_coeff,
                          objective_vars,
                          objective_constant,
                          sense=gurobipy.GRB.MAXIMIZE)
        return BarrierDerivReturn(milp, x, x_next, relu_binary_current,
                                  relu_binary_next, system_constraint_return,
                                  relu_mip_cnstr_return)

    def derivative_loss_at_samples_and_next_states(self,
                                                   x_star,
                                                   c: float,
                                                   epsilon: float,
                                                   state_samples,
                                                   state_next,
                                                   *,
                                                   margin=0.,
                                                   reduction="mean"):
        """
        Take the sample state xⁱ, and compute the total loss on all these
        samples. Each state has a loss max(−hdot(xⁱ)−εh(xⁱ) + margin, 0).
        Args:
          reduction: if reduction=="mean", then we take the average loss among
          all samples; if reduction=="max", then we take the maximal loss among
          all samples; if recution=="4norm", then we take the 4-norm for all
          samples.
        """
        loss_all = torch.nn.HingeEmbeddingLoss(
            margin=margin, reduction="none")(
                -(-self.value(state_next, x_star, c) +
                  (1 - epsilon) * self.value(state_samples, x_star, c)),
                torch.tensor(-1.))
        return utils.loss_reduction(loss_all, reduction)
