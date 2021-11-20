import neural_network_lyapunov.barrier as barrier
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.nominal_controller as nominal_controller

import torch
import gurobipy
import inspect
import time
import wandb


class TrainBarrier:
    """
    Find a barrier function h(x) satisfying
    1. h(x) < 0 for x∈Cᵤ where Cᵤ is an unsafe set.
    2. ḣ(x) ≥ −εh(x) ∀ x∈ℬ whereℬ is a bounded region for verification.
    3. h(x) < 0 ∀ x∈∂ℬ where ∂ℬ is the boundary of the verification region ℬ
    4. h(x*) > 0 for some safe state x*.
    The barrier function is parameterized as h(x)=ϕ(x) − ϕ(x*) + c where ϕ(x)
    is a feedforward neural network with leaky Relu units.

    We will provide two approaches to search for h(x)
    1. Bi-level optimization approach
    2. Adversarial training approach.
    """
    def __init__(self,
                 barrier_system: barrier.Barrier,
                 x_star: torch.Tensor,
                 c: float,
                 unsafe_region_cnstr: gurobi_torch_mip.
                 MixedIntegerConstraintsReturn,
                 verify_region_boundary: gurobi_torch_mip.
                 MixedIntegerConstraintsReturn,
                 epsilon: float,
                 inf_norm_term: barrier.InfNormTerm = None):
        """
        Args:
          barrier_system: a Barrier class object.
          x_star: one safe state.
          c: a positive constant used in the barrier function
          h(x) = ϕ(x) - ϕ(x*) + c.
          unsafe_region_cnstr: The constraints encoding x∈Cᵤ
          verify_region_boundary: The constraints encoding x∈∂ℬ
          inf_norm_term: If not None, then we add the infinity term -|Rx-p|∞
          to the barrier function. This is useful to make h(x) to be negative
          on the boundary of the box region.
        """
        assert (isinstance(barrier_system, barrier.Barrier))
        self.barrier_system = barrier_system
        assert (isinstance(x_star, torch.Tensor))
        assert (x_star.shape == (barrier_system.system.x_dim, ))
        self.x_star = x_star
        assert (isinstance(c, float))
        assert (c > 0)
        self.c = c
        assert (isinstance(unsafe_region_cnstr,
                           gurobi_torch_mip.MixedIntegerConstraintsReturn))
        self.unsafe_region_cnstr = unsafe_region_cnstr
        assert (isinstance(verify_region_boundary,
                           gurobi_torch_mip.MixedIntegerConstraintsReturn))
        self.verify_region_boundary = verify_region_boundary
        assert (isinstance(epsilon, float))
        assert (epsilon > 0)
        self.epsilon = epsilon

        assert (inf_norm_term is None
                or isinstance(inf_norm_term, barrier.InfNormTerm))
        self.inf_norm_term = inf_norm_term

        self.learning_rate = 0.003
        # momentum used in SGD.
        self.momentum = 0.
        self.max_iterations = 1000
        # The weight of the MIP cost hinge(max h(x), x∈Cᵤ)
        self.unsafe_region_mip_cost_weight = 1.
        # The weight of the MIP cost hinge(max h(x), x∈∂ℬ)
        self.verify_region_boundary_mip_cost_weight = 1.
        # The weight of the MIP cost hinge(max -ḣ(x) −εh(x))
        self.barrier_deriv_mip_cost_weight = 1.

        # weight on the loss of the sampled states.
        self.unsafe_state_samples_weight = None
        self.boundary_state_samples_weight = None
        self.derivative_state_samples_weight = None

        # The margin used to penalize the unsafe MIP cost. We use
        # max(mip_cost + margin, 0)
        self.unsafe_mip_margin = 0.
        self.boundary_mip_margin = 0.
        self.deriv_mip_margin = 0.

        # We support Adam or SGD.
        self.optimizer = "Adam"

        self.unsafe_mip_params = {gurobipy.GRB.Param.OutputFlag: False}

        self.verify_region_boundary_mip_params = {
            gurobipy.GRB.Param.OutputFlag: False
        }

        self.barrier_deriv_mip_params = {gurobipy.GRB.Param.OutputFlag: False}

        # Enable wandb to log the data.
        self.enable_wandb = False

        self.output_flag = True

    def _solve_barrier_value_mip(self, region_cnstr, region_name, mip_params):
        ret = self.barrier_system.barrier_value_as_milp(
            self.x_star,
            self.c,
            region_cnstr,
            region_name,
            inf_norm_term=self.inf_norm_term)
        for param, val in mip_params.items():
            ret.milp.gurobi_model.setParam(param, val)
        ret.milp.gurobi_model.optimize()
        assert (ret.milp.gurobi_model.status == gurobipy.GRB.Status.OPTIMAL)
        return ret.milp, ret.x

    def solve_unsafe_region_mip(self):
        return self._solve_barrier_value_mip(self.unsafe_region_cnstr,
                                             "unsafe", self.unsafe_mip_params)

    def solve_verify_region_boundary_mip(self):
        return self._solve_barrier_value_mip(
            self.verify_region_boundary, "verify_boundary",
            self.verify_region_boundary_mip_params)

    def solve_barrier_deriv_mip(self):
        deriv_return = self.barrier_system.barrier_derivative_as_milp(
            self.x_star,
            self.c,
            self.epsilon,
            inf_norm_term=self.inf_norm_term,
            binary_var_type=gurobipy.GRB.BINARY)
        for param, val in self.barrier_deriv_mip_params.items():
            deriv_return.milp.gurobi_model.setParam(param, val)
        deriv_return.milp.gurobi_model.optimize()
        assert (deriv_return.milp.gurobi_model.status ==
                gurobipy.GRB.Status.OPTIMAL)
        return deriv_return.milp, deriv_return.x

    def compute_sample_loss(self, unsafe_state_samples, boundary_state_samples,
                            derivative_state_samples,
                            unsafe_state_samples_weight,
                            boundary_state_samples_weight,
                            derivative_state_samples_weight):
        """
        Compute the sum of these losses.
        1. hinge(h(xⁱ)), xⁱ in unsafe_state_samples
        2. hinge(h(xⁱ)), xⁱ in boundary_state_samples
        3. hinge(−ḣ(xⁱ) − εh(xⁱ)) xⁱ in derivative_state_samples
        """
        total_loss = torch.tensor(0, dtype=self.barrier_system.system.dtype)
        if unsafe_state_samples_weight is not None and\
                unsafe_state_samples.shape[0] > 0:
            h_unsafe = self.barrier_system.barrier_value(
                unsafe_state_samples,
                self.x_star,
                self.c,
                inf_norm_term=self.inf_norm_term)
            total_loss += unsafe_state_samples_weight * \
                torch.nn.HingeEmbeddingLoss(margin=0., reduction="mean")(
                    -h_unsafe, torch.tensor(-1))
        if boundary_state_samples_weight is not None and\
                boundary_state_samples.shape[0] > 0:
            h_boundary = self.barrier_system.barrier_value(
                boundary_state_samples,
                self.x_star,
                self.c,
                inf_norm_term=self.inf_norm_term)
            total_loss += boundary_state_samples_weight * \
                torch.nn.HingeEmbeddingLoss(margin=0., reduction="mean")(
                    -h_boundary, torch.tensor(-1))
        if derivative_state_samples_weight is not None and\
                derivative_state_samples.shape[0] > 0:
            hdot = torch.stack([
                torch.min(
                    self.barrier_system.barrier_derivative(
                        derivative_state_samples[i],
                        inf_norm_term=self.inf_norm_term))
                for i in range(derivative_state_samples.shape[0])
            ])
            h = self.barrier_system.barrier_value(
                derivative_state_samples,
                self.x_star,
                self.c,
                inf_norm_term=self.inf_norm_term)
            total_loss += derivative_state_samples_weight * \
                torch.nn.HingeEmbeddingLoss(margin=0., reduction="mean")(
                    hdot + self.epsilon * h, torch.tensor(-1))
        return total_loss

    class TotalLossReturn:
        def __init__(self, loss, unsafe_mip_objective,
                     verify_region_boundary_mip_objective,
                     barrier_deriv_mip_objective, sample_loss):
            self.loss = loss
            self.unsafe_mip_objective = unsafe_mip_objective
            self.verify_region_boundary_mip_objective = \
                verify_region_boundary_mip_objective
            self.barrier_deriv_mip_objective = barrier_deriv_mip_objective
            self.sample_loss = sample_loss

    def total_loss(self, unsafe_mip_cost_weight: float,
                   verify_region_boundary_mip_cost_weight: float,
                   barrier_deriv_mip_cost_weight: float,
                   unsafe_state_samples: torch.Tensor,
                   boundary_state_samples: torch.Tensor,
                   deriv_state_samples: torch.Tensor) -> TotalLossReturn:
        """
        Compute the total loss as the summation of
        1. hinge(max h(x), x∈Cᵤ)
        2. hinge(max h(x), x∈∂ℬ)
        3. hinge(max −ḣ(x) − εh(x), x∈ℬ)
        4. hinge(h(xⁱ)), xⁱ in unsafe_state_samples
        5. hinge(h(xⁱ)), xⁱ in boundary_state_samples
        6. hinge(−ḣ(xⁱ) − εh(xⁱ)) xⁱ in deriv_state_samples
        """
        dtype = self.barrier_system.system.dtype
        loss = torch.tensor(0, dtype=dtype)
        if unsafe_mip_cost_weight is not None and (
                self.unsafe_region_cnstr.num_ineq() > 0
                or self.unsafe_region_cnstr.num_eq() > 0):
            unsafe_mip, unsafe_x = self.solve_unsafe_region_mip()
            unsafe_mip_objective = unsafe_mip.gurobi_model.ObjVal
            if unsafe_mip_cost_weight > 0:
                loss += unsafe_mip_cost_weight * torch.maximum(
                    torch.tensor(0, dtype=dtype),
                    unsafe_mip.compute_objective_from_mip_data_and_solution(
                        solution_number=0, penalty=1E-13) +
                    self.unsafe_mip_margin)
        else:
            unsafe_mip_objective = None

        if verify_region_boundary_mip_cost_weight is not None:
            verify_region_boundary_mip, verify_region_boundary_x = \
                self.solve_verify_region_boundary_mip()
            verify_region_boundary_mip_objective = \
                verify_region_boundary_mip.gurobi_model.ObjVal
            if verify_region_boundary_mip_cost_weight > 0:
                loss += verify_region_boundary_mip_cost_weight * torch.maximum(
                    torch.tensor(0, dtype=dtype),
                    verify_region_boundary_mip.
                    compute_objective_from_mip_data_and_solution(
                        solution_number=0, penalty=1E-13) +
                    self.boundary_mip_margin)
        else:
            verify_region_boundary_mip_objective = None

        if barrier_deriv_mip_cost_weight is not None:
            barrier_deriv_mip, barrier_deriv_x = self.solve_barrier_deriv_mip()
            barrier_deriv_mip_objective = barrier_deriv_mip.gurobi_model.ObjVal
            if barrier_deriv_mip_cost_weight > 0:
                loss += barrier_deriv_mip_cost_weight * torch.maximum(
                    torch.tensor(0, dtype=dtype),
                    barrier_deriv_mip.
                    compute_objective_from_mip_data_and_solution(
                        solution_number=0, penalty=1E-13) +
                    self.deriv_mip_margin)
        else:
            barrier_deriv_mip_objective = None

        sample_loss = self.compute_sample_loss(
            unsafe_state_samples, boundary_state_samples, deriv_state_samples,
            self.unsafe_state_samples_weight,
            self.boundary_state_samples_weight,
            self.derivative_state_samples_weight)
        loss += sample_loss

        return TrainBarrier.TotalLossReturn(
            loss, unsafe_mip_objective, verify_region_boundary_mip_objective,
            barrier_deriv_mip_objective, sample_loss)

    def print(self):
        for attr in inspect.getmembers(self):
            if not attr[0].startswith('_') and not inspect.ismethod(attr[1]):
                if attr[0] not in ('barrier_system'):
                    print(f"{attr[0]}: {attr[1]}")
                    if self.enable_wandb:
                        wandb.config.update({attr[0]: f"{attr[1]}"})

    def _training_params(self):
        training_params = list(self.barrier_system.barrier_relu.parameters())
        return training_params

    def train(self, unsafe_state_samples, boundary_state_samples,
              deriv_state_samples):
        train_start_time = time.time()
        if self.output_flag:
            self.print()

        iter_count = 0
        training_params = self._training_params()

        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(training_params,
                                         lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(training_params,
                                        lr=self.learning_rate,
                                        momentum=self.momentum)
        else:
            raise Exception(
                "train(): unknown optimizer. Only support Adam or SGD.")

        while iter_count < self.max_iterations:
            optimizer.zero_grad()
            total_loss_return = self.total_loss(
                self.unsafe_region_mip_cost_weight,
                self.verify_region_boundary_mip_cost_weight,
                self.barrier_deriv_mip_cost_weight, unsafe_state_samples,
                boundary_state_samples, deriv_state_samples)
            if self.enable_wandb:
                wandb.log({
                    "loss":
                    total_loss_return.loss.item(),
                    "unsafe_mip_objective":
                    total_loss_return.unsafe_mip_objective,
                    "barrier_deriv_mip_objective":
                    total_loss_return.barrier_deriv_mip_objective,
                    "verify_region_boundary_mip_objective":
                    total_loss_return.verify_region_boundary_mip_objective,
                    "time":
                    time.time() - train_start_time
                })
            if self.output_flag:
                print(
                    f"Iter {iter_count}, " +
                    f"loss {total_loss_return.loss.item()}, " +
                    "unsafe_mip_objective " +
                    f"{total_loss_return.unsafe_mip_objective}, " +
                    "barrier_deriv_mip_objective " +
                    f"{total_loss_return.barrier_deriv_mip_objective}, " +
                    "verify_region_boundary_mip_objective " +
                    f"{total_loss_return.verify_region_boundary_mip_objective}"
                )

            if (total_loss_return.unsafe_mip_objective is None or
                total_loss_return.unsafe_mip_objective <= 0) and\
                total_loss_return.barrier_deriv_mip_objective <= 0 and\
                (total_loss_return.verify_region_boundary_mip_objective is None
                 or total_loss_return.verify_region_boundary_mip_objective <= 0):  # noqa
                return True

            total_loss_return.loss.backward()
            optimizer.step()
            iter_count += 1
        return False

    def train_on_samples(self, unsafe_state_samples, boundary_state_samples,
                         deriv_state_samples):
        """
        Train the barrier function on the samples
        """
        if self.unsafe_state_samples_weight is None or\
            self.boundary_state_samples_weight is None or\
                self.derivative_state_samples_weight is None:
            raise Warning("The sample cost weight is None. Better to set " +
                          "it to a positive scalar.")
        training_params = self._training_params()
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(training_params,
                                         lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(training_params,
                                        lr=self.learning_rate,
                                        momentum=self.momentum)
        else:
            raise Exception(
                "train(): unknown optimizer. Only support Adam or SGD.")

        iter_count = 0
        while iter_count < self.max_iterations:
            optimizer.zero_grad()
            total_loss_return = self.total_loss(None, None, None,
                                                unsafe_state_samples,
                                                boundary_state_samples,
                                                deriv_state_samples)
            if self.output_flag:
                print(f"Iter {iter_count}, " +
                      f"loss {total_loss_return.loss.item()},")
            if total_loss_return.loss < 0:
                return
            total_loss_return.loss.backward()
            optimizer.step()
            iter_count += 1
        return

    def nominal_controller_loss(
            self, controller: nominal_controller.NominalController,
            sample_states: torch.Tensor, weight: float):
        """
        When we train a control barrier function, we can also search for a
        nominal controller that is consistent with this CBF, namely
        ∂h/∂x*(f(x)+G(x)π(x)) ≥ −εh(x) should hold on the sample states. We
        penalize the violation of this condition on the sample states. Adding
        this loss will help the CBF-induced controller to take more smooth
        actions.
        """
        assert (isinstance(controller, nominal_controller.NominalController))
        sample_actions = controller.output(sample_states)
        assert (isinstance(sample_states, torch.Tensor))
        assert (len(sample_states.shape) == 2
                and sample_states.shape[1] == self.barrier_system.system.x_dim)
        sample_loss = -self.epsilon * self.barrier_system.barrier_value(
            sample_states,
            self.x_star,
            self.c,
            inf_norm_term=self.inf_norm_term
        ) - self.barrier_system.minimal_barrier_derivative_given_action(
            sample_states, sample_actions, inf_norm_term=self.inf_norm_term)

        return weight * torch.nn.HingeEmbeddingLoss(
            margin=0., reduction="mean")(-sample_loss, torch.tensor(-1))
