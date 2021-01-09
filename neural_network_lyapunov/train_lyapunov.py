import torch
import torch.utils.tensorboard
import numpy as np
import gurobipy
import copy
import wandb
import inspect
import time
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.utils as utils


class SearchROptions:
    """
    When search for the Lyapunov function, we use the 1-norm of |R*(x-x*)|₁.
    This class specificies the options to search for R.
    """

    def __init__(self, R_size, epsilon):
        """
        We want R to be a full column rank matrix, with size m x n and m >= n.
        The first n rows of R (The square matrix on top of R) is parameterized
        as L * L' + eps * I to make sure the induced 2-norm of R is at least
        eps.
        @param R_size the size of R, with R_size[0] >= R_size[1]
        @param epsilon eps in the documentation above.
        """
        assert(len(R_size) == 2)
        assert(R_size[0] >= R_size[1])
        self.R_size = R_size
        self._variables = torch.empty((
            int(R_size[1] * (R_size[1]+1)/2) + (R_size[0] - R_size[1]) *
            R_size[1],), dtype=torch.float64, requires_grad=True)
        assert(epsilon > 0)
        self.epsilon = epsilon

    def set_variable_value(self, R_val: np.ndarray):
        assert(isinstance(R_val, np.ndarray))
        assert(R_val.shape == self.R_size)
        R_top = R_val[:R_val.shape[1], :]
        R_top = (R_top + R_top.T) / 2
        L = np.linalg.cholesky(R_top - self.epsilon * np.eye(R_val.shape[1]))
        L_entry_count = 0
        variable_val = np.empty((self._variables.shape[0],))
        for i in range(self.R_size[1]):
            variable_val[L_entry_count:L_entry_count+self.R_size[1]-i] =\
                L[i:, i]
            L_entry_count += self.R_size[1] - i
        variable_val[L_entry_count:] = R_val[self.R_size[1]:, :].reshape((-1,))
        self._variables = torch.from_numpy(variable_val)
        self._variables.requires_grad = True

    def R(self):
        L_entry_count = int(self.R_size[1] * (self.R_size[1] + 1) / 2)
        L_lower_list = torch.split(
            self._variables[:L_entry_count], np.arange(
                1, self.R_size[1]+1, 1, dtype=int)[::-1].tolist())
        L_list = []
        for i in range(self.R_size[1]):
            L_list.append(torch.zeros((i,), dtype=torch.float64))
            L_list.append(L_lower_list[i])
        L = torch.cat(L_list).reshape((self.R_size[1], self.R_size[1])).T
        R_bottom = self._variables[L_entry_count:].reshape((
            self.R_size[0]-self.R_size[1], self.R_size[1]))
        R = torch.cat((
            L @ L.T + self.epsilon * torch.eye(
                self.R_size[1], dtype=torch.float64), R_bottom), dim=0)
        return R

    def variables(self):
        return [self._variables]

    @property
    def fixed_R(self):
        return False


class FixedROptions:
    """
    When search for the Lyapunov function, we use the 1-norm of |R*(x-x*)|₁.
    This class specificies that R is fixed.
    R should be fixed to a full column rank matrix.
    """

    def __init__(self, R: torch.Tensor):
        assert(isinstance(R, torch.Tensor))
        self._R = R

    def R(self):
        return self._R

    def variables(self):
        return []

    @property
    def fixed_R(self):
        return True


class TrainLyapunovReLU:
    """
    We will train a ReLU network, such that the function
    V(x) = ReLU(x) - ReLU(x*) + λ|R*(x-x*)|₁ is a Lyapunov function that
    certifies exponential/asymptotic convergence. Namely V(x) should satisfy
    the following conditions
    1. V(x) > 0 ∀ x ≠ x*
    2. dV(x) ≤ -ε V(x) ∀ x for exponential convergence
       dV(x) < 0 for asymptotic convergence
    where dV(x) = V̇(x) for continuous time system, and
    dV(x[n]) = V(x[n+1]) - V(x[n]) for discrete time system.
    In order to find such V(x), we penalize a (weighted) sum of the following
    loss
    1. hinge(-V(xⁱ)) for sampled state xⁱ.
    2. hinge(dV(xⁱ) + ε V(xⁱ)) for sampled state xⁱ.
    3. -min_x V(x) - ε₂ |R*(x - x*)|₁
    4. max_x dV(x) + ε V(x)
    where ε₂ is a given positive scalar, and |R*(x - x*)|₁ is the 1-norm of
    R*(x - x*).
    hinge(z) = max(z + margin, 0) where margin is a given scalar.
    """

    def __init__(
            self, lyapunov_hybrid_system, V_lambda, x_equilibrium, R_options):
        """
        @param lyapunov_hybrid_system This input should define a common
        interface
        lyapunov_positivity_as_milp() (which represents
        minₓ V(x) - ε₂ |R*(x - x*)|₁)
        lyapunov_positivity_loss_at_samples() (which represents
        mean(hinge(-V(xⁱ))))
        lyapunov_derivative_as_milp() (which represents maxₓ dV(x) + ε V(x))
        lyapunov_derivative_loss_at_samples() (which represents
        mean(hinge(dV(xⁱ) + ε V(xⁱ))).
        One example of input type is lyapunov.LyapunovDiscreteTimeHybridSystem.
        @param V_lambda λ in the documentation above.
        @param x_equilibrium The equilibrium state.
        @param R_options Either SearchROptions or FixedROptions.
        """
        self.lyapunov_hybrid_system = lyapunov_hybrid_system
        assert(isinstance(V_lambda, float))
        self.V_lambda = V_lambda
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (lyapunov_hybrid_system.system.x_dim,))
        self.x_equilibrium = x_equilibrium
        assert(isinstance(R_options, SearchROptions) or
               isinstance(R_options, FixedROptions))
        self.R_options = R_options
        # The learning rate of the optimizer
        self.learning_rate = 0.003
        # Number of iterations in the training.
        self.max_iterations = 1000
        # The weight of hinge(-V(xⁱ)) in the total loss
        self.lyapunov_positivity_sample_cost_weight = 1.
        # The margin in hinge(-V(xⁱ))
        self.lyapunov_positivity_sample_margin = 0.
        # The weight of hinge(dV(xⁱ) + ε V(xⁱ)) in the total loss
        self.lyapunov_derivative_sample_cost_weight = 1.
        # The margin in hinge(dV(xⁱ) + ε V(xⁱ))
        self.lyapunov_derivative_sample_margin = 0.
        # The weight of -min_x V(x) - ε₂ |x - x*|₁ in the total loss
        self.lyapunov_positivity_mip_cost_weight = 10.
        # The number of (sub)optimal solutions for the MIP
        # min_x V(x) - V(x*) - ε₂ |x - x*|₁
        self.lyapunov_positivity_mip_pool_solutions = 1
        # The weight of max_x dV(x) + εV(x)
        self.lyapunov_derivative_mip_cost_weight = 10.
        # The number of (sub)optimal solutions for the MIP
        # max_x dV(x) + εV(x)
        self.lyapunov_derivative_mip_pool_solutions = 1
        # If set to true, we will print some messages during training.
        self.output_flag = False
        # The MIP objective loss is ∑ⱼ rʲ * j_th_objective. r is the decay
        # rate.
        self.lyapunov_positivity_mip_cost_decay_rate = 0.9
        self.lyapunov_derivative_mip_cost_decay_rate = 0.9
        # This is ε₂ in  V(x) >=  ε₂ |x - x*|₁
        self.lyapunov_positivity_epsilon = 0.01
        # This is ε in dV(x) ≤ -ε V(x) or dV(x) ≤ -ε |x-x*|₁
        self.lyapunov_derivative_epsilon = 0.01
        # Depending on this type, the interpretation for
        # self.lyapunov_derivative_epsilon is different. To prove exponential
        # convergence, use ExpLower (or ExpUpper) to prove the exponential
        # convergence rate. To prove asymptotic convergence, use Asymp
        self.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower

        # The convergence tolerance for the training.
        # When the lyapunov function
        # minₓ V(x)-ε₂|x-x*|₁ > -lyapunov_positivity_convergence_tol
        # and the lyapunov derivative satisfies
        # maxₓ dV(x) + εV(x) < lyapunov_derivative_convergence_tol
        # we regard that the Lyapunov function is found.
        self.lyapunov_positivity_convergence_tol = 1E-6
        self.lyapunov_derivative_convergence_tol = 3e-5

        # We support Adam or SGD.
        self.optimizer = "Adam"

        # If summary writer is not None, then we use tensorboard to write
        # training loss to the summary writer.
        self.summary_writer_folder = None

        # Enable wandb to log the data.
        self.enable_wandb = False

        # Save the neural network model to this folder. If set to None, then
        # don't save the network.
        self.save_network_path = None
        # By default, every 10 iterations in the optimizer, we save the
        # network.
        self.save_network_iterations = 10

        # parameter used in SGD
        self.momentum = 0.

        # Whether we add the positivity condition adversarial states to the
        # training set.
        self.add_positivity_adversarial_state = False
        # Whether we add the derivative condition adversarial states to the
        # training set.
        self.add_derivative_adversarial_state = False
        # We compute the sample loss on the most recent max_sample_pool_size
        # samples.
        self.max_sample_pool_size = 500

        # Whether we search for the controller when the dynamical is a feedback
        # system which contains a neural network representing its controller.
        self.search_controller = True

        # Whether to search over R when we use the 1-norm of |R*(x-x*)|₁.
        self.search_R = False

        # All the Lyapunov derivative MIP params (except pool solutions).
        self.lyapunov_derivative_mip_params = {
            gurobipy.GRB.Param.OutputFlag: False}

        # Early termination: if None, will
        # find the most adversarial state, otherwise will terminate the search
        # as soon as it finds a state that violates the positivity or
        # derivative condition by at least the given parameter
        self.lyapunov_positivity_mip_term_threshold = None
        self.lyapunov_derivative_mip_term_threshold = None

        # Warmstart: Boolean as to whether or not reuse
        # the last adversarial samples as warmstart for the current loss
        # evaluation
        self.lyapunov_positivity_mip_warmstart = False
        self.lyapunov_derivative_mip_warmstart = False
        self.lyapunov_positivity_last_x_adv = None
        self.lyapunov_derivative_last_x_adv = None

    def total_loss(
            self, positivity_state_samples, derivative_state_samples,
            derivative_state_samples_next,
            lyapunov_positivity_sample_cost_weight,
            lyapunov_derivative_sample_cost_weight,
            lyapunov_positivity_mip_cost_weight,
            lyapunov_derivative_mip_cost_weight):
        """
        Compute the total loss as the summation of
        1. hinge(-V(xⁱ) + ε₂ |xⁱ - x*|₁) for sampled state xⁱ.
        2. hinge(dV(xⁱ) + ε V(xⁱ)) for sampled state xⁱ.
        3. -min_x V(x) - ε₂ |x - x*|₁
        4. max_x dV(x) + ε V(x)
        @param positivity_state_samples All sample states on which we compute
        the violation of Lyapunov positivity constraint.
        @param derivative_state_samples All sample states on which we compute
        the violation of Lyapunov derivative constraint.
        @param derivative_state_samples_next. The next state(s) of the sampled
        state. derivative_state_samples_next contains next state(s) of
        derivative_state_samples[i].
        @param loss, positivity_mip_objective, derivative_mip_objective,
        positivity_sample_loss, derivative_sample_loss, positivity_mip_loss,
        derivative_mip_loss
        positivity_mip_objective is the objective value
        min_x V(x) - ε₂ |x - x*|₁. We want this value to be non-negative.
        derivative_mip_objective is the objective value max_x dV(x) + ε V(x).
        We want this value to be non-positive.
        positivity_sample_loss is weight * cost1
        derivative_sample_loss is weight * cost2
        positivity_mip_loss is weight * cost3
        derivative_mip_loss is weight * cost4
        """
        assert(isinstance(positivity_state_samples, torch.Tensor))
        assert(isinstance(derivative_state_samples, torch.Tensor))
        assert(isinstance(derivative_state_samples_next, torch.Tensor))
        assert(
            positivity_state_samples.shape[1] ==
            self.lyapunov_hybrid_system.system.x_dim)
        assert(
            derivative_state_samples.shape[1] ==
            self.lyapunov_hybrid_system.system.x_dim)
        assert(
            derivative_state_samples_next.shape ==
            (derivative_state_samples.shape[0],
             self.lyapunov_hybrid_system.system.x_dim))
        dtype = self.lyapunov_hybrid_system.system.dtype
        if lyapunov_positivity_mip_cost_weight is not None:
            lyapunov_positivity_as_milp_return = self.lyapunov_hybrid_system.\
                lyapunov_positivity_as_milp(
                    self.x_equilibrium, self.V_lambda,
                    self.lyapunov_positivity_epsilon, R=self.R_options.R(),
                    fixed_R=self.R_options.fixed_R,
                    x_warmstart=self.lyapunov_positivity_last_x_adv)
            lyapunov_positivity_mip = lyapunov_positivity_as_milp_return[0]
            lyapunov_positivity_mip.gurobi_model.setParam(
                gurobipy.GRB.Param.OutputFlag, False)
            if self.lyapunov_positivity_mip_pool_solutions > 1:
                lyapunov_positivity_mip.gurobi_model.setParam(
                    gurobipy.GRB.Param.PoolSearchMode, 2)
                lyapunov_positivity_mip.gurobi_model.setParam(
                    gurobipy.GRB.Param.PoolSolutions,
                    self.lyapunov_positivity_mip_pool_solutions)
            if self.lyapunov_positivity_mip_term_threshold is not None:
                lyapunov_positivity_mip.gurobi_model.optimize(
                    utils.get_gurobi_terminate_if_callback(
                        threshold=self.lyapunov_positivity_mip_term_threshold))
            else:
                lyapunov_positivity_mip.gurobi_model.optimize()
            lyapunov_positivity_mip_obj = \
                lyapunov_positivity_mip.gurobi_model.ObjVal
            if self.lyapunov_positivity_mip_warmstart:
                self.lyapunov_positivity_last_x_adv = torch.tensor([
                    v.x for v in lyapunov_positivity_as_milp_return[1]],
                    dtype=self.lyapunov_hybrid_system.system.dtype)
        else:
            lyapunov_positivity_mip_obj = np.nan

        if lyapunov_derivative_mip_cost_weight is not None:
            lyapunov_derivative_as_milp_return = self.lyapunov_hybrid_system.\
                lyapunov_derivative_as_milp(
                    self.x_equilibrium, self.V_lambda,
                    self.lyapunov_derivative_epsilon,
                    self.lyapunov_derivative_eps_type, R=self.R_options.R(),
                    fixed_R=self.R_options.fixed_R,
                    x_warmstart=self.lyapunov_derivative_last_x_adv)
            lyapunov_derivative_mip = lyapunov_derivative_as_milp_return[0]
            for param, val in self.lyapunov_derivative_mip_params.items():
                lyapunov_derivative_mip.gurobi_model.setParam(param, val)
            if (self.lyapunov_derivative_mip_pool_solutions > 1):
                lyapunov_derivative_mip.gurobi_model.setParam(
                    gurobipy.GRB.Param.PoolSearchMode, 2)
                lyapunov_derivative_mip.gurobi_model.setParam(
                    gurobipy.GRB.Param.PoolSolutions,
                    self.lyapunov_derivative_mip_pool_solutions)
            if self.lyapunov_derivative_mip_term_threshold is not None:
                lyapunov_derivative_mip.gurobi_model.optimize(
                    utils.get_gurobi_terminate_if_callback(
                        threshold=self.lyapunov_derivative_mip_term_threshold))
            else:
                lyapunov_derivative_mip.gurobi_model.optimize()
            lyapunov_derivative_mip_obj = \
                lyapunov_derivative_mip.gurobi_model.ObjVal

            relu_zeta_val = np.array([
                np.round(v.x) for v in lyapunov_derivative_as_milp_return[2]])
            if self.output_flag:
                print("lyapunov derivative MIP Relu activation: "
                      f"{np.argwhere(relu_zeta_val == 1).squeeze()}")
                print(
                    "adversarial x " +
                    f"{[v.x for v in lyapunov_derivative_as_milp_return[1]]}")
            if self.lyapunov_derivative_mip_warmstart:
                self.lyapunov_derivative_last_x_adv = torch.tensor([
                    v.x for v in lyapunov_derivative_as_milp_return[1]],
                    dtype=self.lyapunov_hybrid_system.system.dtype)
        else:
            lyapunov_derivative_mip_obj = np.nan

        loss = torch.tensor(0., dtype=dtype)
        relu_at_equilibrium = \
            self.lyapunov_hybrid_system.lyapunov_relu.forward(
                self.x_equilibrium)

        positivity_mip_loss = 0.
        if lyapunov_positivity_mip_cost_weight != 0 and\
                lyapunov_positivity_mip_cost_weight is not None:
            for mip_sol_number in range(
                    self.lyapunov_positivity_mip_pool_solutions):
                if mip_sol_number < \
                        lyapunov_positivity_mip.gurobi_model.solCount:
                    positivity_mip_loss += \
                        lyapunov_positivity_mip_cost_weight * \
                        torch.pow(torch.tensor(
                            self.lyapunov_positivity_mip_cost_decay_rate,
                            dtype=dtype), mip_sol_number) *\
                        lyapunov_positivity_mip.\
                        compute_objective_from_mip_data_and_solution(
                            solution_number=mip_sol_number, penalty=1e-13)
        derivative_mip_loss = 0
        if lyapunov_derivative_mip_cost_weight != 0\
                and lyapunov_derivative_mip_cost_weight is not None:
            for mip_sol_number in range(
                    self.lyapunov_derivative_mip_pool_solutions):
                if (mip_sol_number <
                        lyapunov_derivative_mip.gurobi_model.solCount):
                    mip_cost = lyapunov_derivative_mip.\
                        compute_objective_from_mip_data_and_solution(
                            solution_number=mip_sol_number, penalty=1e-13)
                    derivative_mip_loss += \
                        lyapunov_derivative_mip_cost_weight *\
                        torch.pow(torch.tensor(
                            self.lyapunov_derivative_mip_cost_decay_rate,
                            dtype=dtype), mip_sol_number) * mip_cost
                    lyapunov_derivative_mip.gurobi_model.setParam(
                        gurobipy.GRB.Param.SolutionNumber, mip_sol_number)

        # We add the most adverisal states of the positivity MIP and derivative
        # MIP to the training set. Note we solve positivity MIP and derivative
        # MIP separately. This is different from adding the most adversarial
        # state of the total loss.
        if self.add_positivity_adversarial_state and \
                lyapunov_positivity_mip_cost_weight is not None:
            positivity_mip_adversarial = torch.tensor([
                v.x for v in lyapunov_positivity_as_milp_return[1]],
                dtype=self.lyapunov_hybrid_system.system.dtype)
            positivity_state_samples = torch.cat(
                [positivity_state_samples,
                 positivity_mip_adversarial.unsqueeze(0)], dim=0)
        if self.add_derivative_adversarial_state and \
                lyapunov_derivative_mip_cost_weight is not None:
            derivative_mip_adversarial = torch.tensor([
                v.x for v in lyapunov_derivative_as_milp_return[1]],
                dtype=self.lyapunov_hybrid_system.system.dtype)
            if isinstance(self.lyapunov_hybrid_system.system,
                          hybrid_linear_system.AutonomousHybridLinearSystem):
                derivative_mip_adversarial_mode = np.argwhere(np.array(
                    [v.x for v in lyapunov_derivative_as_milp_return[3]])
                    > 0.99)[0][0]
                derivative_mip_adversarial_next = self.lyapunov_hybrid_system.\
                    system.step_forward(
                        derivative_mip_adversarial,
                        derivative_mip_adversarial_mode)
            else:
                derivative_mip_adversarial_next = self.lyapunov_hybrid_system.\
                    system.step_forward(derivative_mip_adversarial)
            derivative_state_samples = torch.cat(
                [derivative_state_samples,
                 derivative_mip_adversarial.unsqueeze(0)], dim=0)
            derivative_state_samples_next = torch.cat(
                [derivative_state_samples_next,
                 derivative_mip_adversarial_next.unsqueeze(0)], dim=0)

        if lyapunov_positivity_sample_cost_weight != 0 and\
                positivity_state_samples.shape[0] > 0:
            if positivity_state_samples.shape[0] > self.max_sample_pool_size:
                positivity_state_samples_in_pool = \
                    positivity_state_samples[-self.max_sample_pool_size:]
            else:
                positivity_state_samples_in_pool = positivity_state_samples
            positivity_sample_loss = lyapunov_positivity_sample_cost_weight *\
                self.lyapunov_hybrid_system.\
                lyapunov_positivity_loss_at_samples(
                    relu_at_equilibrium, self.x_equilibrium,
                    positivity_state_samples_in_pool, self.V_lambda,
                    self.lyapunov_positivity_epsilon, R=self.R_options.R(),
                    margin=self.lyapunov_positivity_sample_margin)
        else:
            positivity_sample_loss = 0.
        if lyapunov_derivative_sample_cost_weight != 0 and\
                derivative_state_samples.shape[0] > 0:
            if derivative_state_samples.shape[0] > self.max_sample_pool_size:
                derivative_state_samples_in_pool = \
                    derivative_state_samples[-self.max_sample_pool_size:]
                derivative_state_samples_next_in_pool = \
                    derivative_state_samples_next[-self.max_sample_pool_size:]
            else:
                derivative_state_samples_in_pool = derivative_state_samples
                derivative_state_samples_next_in_pool = \
                    derivative_state_samples_next
            derivative_sample_loss = lyapunov_derivative_sample_cost_weight *\
                self.lyapunov_hybrid_system.\
                lyapunov_derivative_loss_at_samples_and_next_states(
                    self.V_lambda, self.lyapunov_derivative_epsilon,
                    derivative_state_samples_in_pool,
                    derivative_state_samples_next_in_pool, self.x_equilibrium,
                    self.lyapunov_derivative_eps_type, R=self.R_options.R(),
                    margin=self.lyapunov_derivative_sample_margin)
        else:
            derivative_sample_loss = 0.

        loss = positivity_sample_loss + derivative_sample_loss + \
            positivity_mip_loss + derivative_mip_loss

        return loss, lyapunov_positivity_mip_obj, lyapunov_derivative_mip_obj,\
            positivity_sample_loss, derivative_sample_loss,\
            positivity_mip_loss, derivative_mip_loss,\
            positivity_state_samples, derivative_state_samples,\
            derivative_state_samples_next

    def _save_network(self, iter_count):
        if self.save_network_path:
            if iter_count % self.save_network_iterations == 0:
                torch.save(
                    self.lyapunov_hybrid_system.lyapunov_relu,
                    self.save_network_path + f'{"/lyapunov.pt"}')
                torch.save(self.R_options.R(),
                           self.save_network_path + f'{"/R.pt"}')
                if isinstance(self.lyapunov_hybrid_system.system,
                              feedback_system.FeedbackSystem):
                    torch.save(
                        self.lyapunov_hybrid_system.system.controller_network,
                        self.save_network_path + f'{"/controller.pt"}')

    def print(self):
        """
        Print the settings of this training device.
        """
        for attr in inspect.getmembers(self):
            if not attr[0].startswith('_') and not inspect.ismethod(attr[1]):
                if attr[0] not in ('lyapunov_hybrid_system', 'R_options',
                                   'summary_writer_folder'):
                    print(attr)

    def train(self, state_samples_all):
        train_start_time = time.time()
        if self.output_flag:
            self.print()
        assert(isinstance(state_samples_all, torch.Tensor))
        assert(
            state_samples_all.shape[1] ==
            self.lyapunov_hybrid_system.system.x_dim)
        positivity_state_samples = state_samples_all.clone()
        derivative_state_samples = state_samples_all.clone()
        if (state_samples_all.shape[0] > 0):
            derivative_state_samples_next = torch.stack([
                self.lyapunov_hybrid_system.system.step_forward(
                    derivative_state_samples[i]) for i in
                range(derivative_state_samples.shape[0])], dim=0)
        else:
            derivative_state_samples_next = torch.empty_like(state_samples_all)
        iter_count = 0
        if isinstance(
            self.lyapunov_hybrid_system.system,
                feedback_system.FeedbackSystem) and self.search_controller:
            # For a feedback system, we train both the Lyapunov network
            # parameters and the controller network parameters.
            training_params = list(
                self.lyapunov_hybrid_system.lyapunov_relu.parameters()) + list(
                    self.lyapunov_hybrid_system.system.controller_network.
                    parameters()) + self.R_options.variables()
        else:
            training_params = \
                list(self.lyapunov_hybrid_system.lyapunov_relu.parameters()) +\
                self.R_options.variables()

        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                training_params, lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                training_params, lr=self.learning_rate, momentum=self.momentum)
        else:
            raise Exception(
                "train: unknown optimizer, only support Adam or SGD.")
        if self.summary_writer_folder is not None:
            writer = torch.utils.tensorboard.SummaryWriter(
                self.summary_writer_folder)
        while iter_count < self.max_iterations:
            self._save_network(iter_count)
            optimizer.zero_grad()
            if isinstance(self.lyapunov_hybrid_system.system,
                          feedback_system.FeedbackSystem):
                # If we train a feedback system, then we will modify the
                # controller in each iteration, hence the next sample state
                # changes in each iteration.
                if (derivative_state_samples.shape[0] > 0):
                    derivative_state_samples_next =\
                        self.lyapunov_hybrid_system.system.step_forward(
                            derivative_state_samples)
                else:
                    derivative_state_samples_next = torch.empty_like(
                        derivative_state_samples)
            loss, lyapunov_positivity_mip_cost,\
                lyapunov_derivative_mip_cost, \
                positivity_sample_loss, derivative_sample_loss,\
                positivity_mip_loss, derivative_mip_loss,\
                positivity_state_samples, derivative_state_samples,\
                derivative_state_samples_next\
                = self.total_loss(
                    positivity_state_samples, derivative_state_samples,
                    derivative_state_samples_next,
                    self.lyapunov_positivity_sample_cost_weight,
                    self.lyapunov_derivative_sample_cost_weight,
                    self.lyapunov_positivity_mip_cost_weight,
                    self.lyapunov_derivative_mip_cost_weight)

            if self.enable_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "positivity MIP cost": lyapunov_positivity_mip_cost,
                    "derivative MIP cost":
                    lyapunov_derivative_mip_cost,
                    "time": time.time() - train_start_time})
            if self.summary_writer_folder is not None:
                writer.add_scalar("loss", loss.item(), iter_count)
                writer.add_scalar(
                    "positivity MIP cost",
                    lyapunov_positivity_mip_cost, iter_count)
                writer.add_scalar(
                    "derivative MIP cost",
                    lyapunov_derivative_mip_cost, iter_count)
            if self.output_flag:
                print(f"Iter {iter_count}, loss {loss}, " +
                      "positivity cost " +
                      f"{lyapunov_positivity_mip_cost}, " +
                      "derivative_cost " +
                      f"{lyapunov_derivative_mip_cost}")
            if lyapunov_positivity_mip_cost <=\
                self.lyapunov_positivity_convergence_tol and\
                lyapunov_derivative_mip_cost <= \
                    self.lyapunov_derivative_convergence_tol:
                return (True, loss.item(),
                        lyapunov_positivity_mip_cost,
                        lyapunov_derivative_mip_cost)
            loss.backward()
            optimizer.step()
            iter_count += 1
        return (False, loss.item(), lyapunov_positivity_mip_cost,
                lyapunov_derivative_mip_cost)

    def train_lyapunov_on_samples(
            self, state_samples_all, num_epochs, batch_size):
        """
        Train a ReLU network on given state samples (not the adversarial states
        found by MIP). The loss function is the weighted sum of the lyapunov
        positivity condition violation and the lyapunov derivative condition
        violation on these samples. We stop the training when either the
        maximum iteration is reached, or when many consecutive iterations the
        MIP costs keeps increasing (which means the network overfits to the
        training data). Return the best network (the one with the minimal
        MIP loss) found so far.
        @param state_samples_all A torch tensor, state_samples_all[i] is the
        i'th sample
        @param max_iterations The maximal number of iterations.
        @param max_increasing_iterations If the MIP loss keeps increasing for
        this number of iterations, stop the training.
        """
        assert(isinstance(state_samples_all, torch.Tensor))
        assert(state_samples_all.shape[1] ==
               self.lyapunov_hybrid_system.system.x_dim)
        best_loss = np.inf
        if isinstance(
            self.lyapunov_hybrid_system.system,
                feedback_system.FeedbackSystem) and self.search_controller:
            training_params = list(
                self.lyapunov_hybrid_system.lyapunov_relu.parameters()) + list(
                    self.lyapunov_hybrid_system.system.controller_network.
                    parameters()) + self.R_options.variables()
        else:
            training_params = \
                list(self.lyapunov_hybrid_system.lyapunov_relu.parameters()) +\
                self.R_options.variables()
        optimizer = torch.optim.Adam(training_params, lr=self.learning_rate)
        dataset = torch.utils.data.TensorDataset(state_samples_all)
        train_set_size = int(len(dataset) * 0.8)
        test_set_size = len(dataset) - train_set_size
        train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_set_size, test_set_size])
        data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_state_samples = test_dataset[:][0]
        for epoch in range(num_epochs):
            running_loss = 0.
            for _, batch_data in enumerate(data_loader):
                state_samples_batch = batch_data[0]
                optimizer.zero_grad()
                state_samples_next = torch.stack([
                    self.lyapunov_hybrid_system.system.step_forward(
                        state_samples_batch[i]) for i in
                    range(state_samples_batch.shape[0])], dim=0)
                loss, lyapunov_positivity_mip_cost,\
                    lyapunov_derivative_mip_cost,  _, _, _, _, _, _, _\
                    = self.total_loss(
                        state_samples_batch, state_samples_batch,
                        state_samples_next,
                        self.lyapunov_positivity_sample_cost_weight,
                        self.lyapunov_derivative_sample_cost_weight,
                        lyapunov_positivity_mip_cost_weight=None,
                        lyapunov_derivative_mip_cost_weight=None)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Compute the test loss
            test_state_samples_next = torch.stack([
                self.lyapunov_hybrid_system.system.step_forward(
                    test_state_samples[i]) for i in
                range(test_state_samples.shape[0])], dim=0)
            test_loss, _, _, _, _, _, _, _, _, _ = self.total_loss(
                test_state_samples, test_state_samples,
                test_state_samples_next,
                self.lyapunov_positivity_sample_cost_weight,
                self.lyapunov_derivative_sample_cost_weight,
                lyapunov_positivity_mip_cost_weight=None,
                lyapunov_derivative_mip_cost_weight=None)
            print(f"epoch {epoch}, training loss " +
                  f"{running_loss / len(data_loader)}, test loss " +
                  f"{test_loss.item()}")
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                best_lyapunov_relu = copy.deepcopy(
                    self.lyapunov_hybrid_system.lyapunov_relu)
                if isinstance(self.lyapunov_hybrid_system.system,
                              feedback_system.FeedbackSystem):
                    best_controller_relu = copy.deepcopy(
                        self.lyapunov_hybrid_system.system.controller_network)

        print(f"best loss {best_loss}")
        self.lyapunov_hybrid_system.lyapunov_relu.load_state_dict(
            best_lyapunov_relu.state_dict())
        if isinstance(self.lyapunov_hybrid_system.system,
                      feedback_system.FeedbackSystem):
            self.lyapunov_hybrid_system.system.controller_network.\
                load_state_dict(best_controller_relu.state_dict())


class TrainValueApproximator:
    """
    Given a piecewise affine system and some sampled initial state, compute the
    cost-to-go for these sampled states, and then train a network to
    approximate the cost-to-go, such that
    network(x) - network(x*) + λ|x-x*|₁ ≈ cost_to_go(x)
    """
    def __init__(self):
        self.max_epochs = 100
        self.convergence_tolerance = 1e-3
        self.learning_rate = 0.02

    def train_with_cost_to_go(
            self, network, x0_value_samples, V_lambda, x_equilibrium, R):
        """
        Similar to train() function, but with given samples on initial_state
        and cost-to-go.
        """
        if R is None:
            x_dim = x_equilibrium.numel()
            R = torch.eye(x_dim, dtype=torch.float64)
        state_samples_all = torch.stack([
            pair[0] for pair in x0_value_samples], dim=0)
        value_samples_all = torch.stack([pair[1] for pair in x0_value_samples])
        optimizer = torch.optim.Adam(
            network.parameters(), lr=self.learning_rate)
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            relu_output = network(state_samples_all)
            relu_x_equilibrium = network.forward(x_equilibrium)
            value_relu = relu_output.squeeze() - relu_x_equilibrium +\
                V_lambda * torch.norm(R @ (
                    state_samples_all - x_equilibrium).T, dim=0, p=1)
            loss = torch.nn.MSELoss()(value_relu, value_samples_all)
            if (loss.item() <= self.convergence_tolerance):
                return True, loss.item()
            loss.backward()
            optimizer.step()
        return False, loss.item()

    def train(
        self, system, network, V_lambda, x_equilibrium, instantaneous_cost_fun,
            x0_samples, T, discrete_time_flag, R, x_goal=None, pruner=None):
        """
        Train a network such that
        network(x) - network(x*) + λ*|x-x*|₁ ≈ cost_to_go(x)
        @param system An AutonomousHybridLinearSystem instance.
        @param network a pytorch neural network.
        @param V_lambda λ.
        @param x_equilibrium x*.
        @param instantaneous_cost_fun A callable to evaluate the instantaneous
        cost for a state.
        @param x0_samples A list of torch tensors. x0_samples[i] is the i'th
        sampled x0.
        @param T The horizon to evaluate the cost-to-go. For a discrete time
        system, T must be an integer, for a continuous time system, T must be a
        float.
        @param discrete_time_flag Whether the hybrid linear system is discrete
        or continuous time.
        @param x_goal The goal position to stop simulating the trajectory when
        computing cost-to-go. Check compute_continuous_time_system_cost_to_go()
        and compute_discrete_time_system_cost_to_go() for more details.
        @param pruner A callable to decide whether to keep a state in the
        training set or not. Check generate_cost_to_go_samples() for more
        details.
        """
        assert(isinstance(
            system, hybrid_linear_system.AutonomousHybridLinearSystem))
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (system.x_dim,))
        assert(isinstance(V_lambda, float))
        assert(isinstance(x0_samples, torch.Tensor))
        x0_value_samples = hybrid_linear_system.generate_cost_to_go_samples(
            system, [x0_samples[i] for i in range(x0_samples.shape[0])], T,
            instantaneous_cost_fun, discrete_time_flag, x_goal, pruner)
        return self.train_with_cost_to_go(
            network, x0_value_samples, V_lambda, x_equilibrium, R)
