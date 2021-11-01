import neural_network_lyapunov.barrier as barrier
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

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
    def __init__(self, barrier_system: barrier.Barrier, x_star: torch.Tensor,
                 c: float, unsafe_region_cnstr: gurobi_torch_mip.
                 MixedIntegerConstraintsReturn,
                 verify_region_boundary: gurobi_torch_mip.
                 MixedIntegerConstraintsReturn, epsilon: float):
        """
        Args:
          barrier_system: a Barrier class object.
          x_star: one safe state.
          c: a positive constant used in the barrier function
          h(x) = ϕ(x) - ϕ(x*) + c.
          unsafe_region_cnstr: The constraints encoding x∈Cᵤ
          verify_region_boundary: The constraints encoding x∈∂ℬ
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
        milp, x = self.barrier_system.barrier_value_as_milp(
            self.x_star, self.c, region_cnstr, region_name)
        for param, val in mip_params.items():
            milp.gurobi_model.setParam(param, val)
        milp.gurobi_model.optimize()
        assert (milp.gurobi_model.status == gurobipy.GRB.Status.OPTIMAL)
        return milp, x

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
            binary_var_type=gurobipy.GRB.BINARY)
        for param, val in self.barrier_deriv_mip_params.items():
            deriv_return.milp.gurobi_model.setParam(param, val)
        deriv_return.milp.gurobi_model.optimize()
        assert (deriv_return.milp.gurobi_model.status ==
                gurobipy.GRB.Status.OPTIMAL)
        return deriv_return.milp, deriv_return.x

    class TotalLossReturn:
        def __init__(self, loss, unsafe_mip_objective,
                     verify_region_boundary_mip_objective,
                     barrier_deriv_mip_objective):
            self.loss = loss
            self.unsafe_mip_objective = unsafe_mip_objective
            self.verify_region_boundary_mip_objective = \
                verify_region_boundary_mip_objective
            self.barrier_deriv_mip_objective = barrier_deriv_mip_objective

    def total_loss(self, unsafe_mip_cost_weight,
                   verify_region_boundary_mip_cost_weight,
                   barrier_deriv_mip_cost_weight) -> TotalLossReturn:
        """
        Compute the total loss as the summation of
        1. hinge(max h(x), x∈Cᵤ)
        2. hinge(max h(x), x∈∂ℬ)
        3. hinge(max −ḣ(x) − εh(x), x∈ℬ)
        """
        dtype = self.barrier_system.system.dtype
        loss = torch.tensor(0, dtype=dtype)
        unsafe_mip, unsafe_x = self.solve_unsafe_region_mip()
        unsafe_mip_objective = unsafe_mip.gurobi_model.ObjVal

        if unsafe_mip_cost_weight > 0:
            loss += unsafe_mip_cost_weight * torch.maximum(
                torch.tensor(0, dtype=dtype),
                unsafe_mip.compute_objective_from_mip_data_and_solution(
                    solution_number=0, penalty=1E-13))

        verify_region_boundary_mip, verify_region_boundary_x = \
            self.solve_verify_region_boundary_mip()
        verify_region_boundary_mip_objective = \
            verify_region_boundary_mip.gurobi_model.ObjVal
        if verify_region_boundary_mip_cost_weight > 0:
            loss += verify_region_boundary_mip_cost_weight * torch.maximum(
                torch.tensor(0, dtype=dtype),
                verify_region_boundary_mip.
                compute_objective_from_mip_data_and_solution(solution_number=0,
                                                             penalty=1E-13))

        barrier_deriv_mip, barrier_deriv_x = self.solve_barrier_deriv_mip()
        barrier_deriv_mip_objective = barrier_deriv_mip.gurobi_model.ObjVal
        if barrier_deriv_mip_cost_weight > 0:
            loss += barrier_deriv_mip_cost_weight * torch.maximum(
                torch.tensor(0, dtype=dtype),
                barrier_deriv_mip.compute_objective_from_mip_data_and_solution(
                    solution_number=0, penalty=1E-13))

        return TrainBarrier.TotalLossReturn(
            loss, unsafe_mip_objective, verify_region_boundary_mip_objective,
            barrier_deriv_mip_objective)

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

    def train(self, state_samples_all: torch.Tensor):
        train_start_time = time.time()
        if self.output_flag:
            self.print()

        assert (isinstance(state_samples_all, torch.Tensor))
        assert (state_samples_all.shape[1] == self.barrier_system.system.x_dim)

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
                self.barrier_deriv_mip_cost_weight)
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
                    f"Iter {iter_count}," +
                    f"loss {total_loss_return.loss.item()}," +
                    "unsafe_mip_objective " +
                    f"{total_loss_return.unsafe_mip_objective}," +
                    "barrier_deriv_mip_objective " +
                    f"{total_loss_return.barrier_deriv_mip_objective}," +
                    "verify_region_boundary_mip_objective " +
                    f"{total_loss_return.verify_region_boundary_mip_objective}"
                )

            if total_loss_return.unsafe_mip_objective <= 0 and\
                total_loss_return.barrier_deriv_mip_objective <= 0 and\
                    total_loss_return.verify_region_boundary_mip_objective <= 0:  # noqa
                return True

            total_loss_return.loss.backward()
            optimizer.step()
            iter_count += 1
        return False