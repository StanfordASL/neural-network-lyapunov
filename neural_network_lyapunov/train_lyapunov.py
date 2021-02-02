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
import neural_network_lyapunov.r_options as r_options


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
    def __init__(self, lyapunov_hybrid_system, V_lambda, x_equilibrium,
                 R_options):
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
        @param R_options An ROptions object.
        """
        self.lyapunov_hybrid_system = lyapunov_hybrid_system
        assert (isinstance(V_lambda, float))
        self.V_lambda = V_lambda
        assert (isinstance(x_equilibrium, torch.Tensor))
        assert (x_equilibrium.shape == (lyapunov_hybrid_system.system.x_dim, ))
        self.x_equilibrium = x_equilibrium
        assert (isinstance(R_options, r_options.ROptions))
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

        # All the Lyapunov derivative MIP params (except pool solutions).
        self.lyapunov_derivative_mip_params = {
            gurobipy.GRB.Param.OutputFlag: False
        }

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

        # If you require only a subset of the states to converge, then set
        # xbar_indices to the indices of these converging states. By default
        # xbar_indices=None means all the states are required to converge.
        # This will set x̂ in the Lyapunov function.
        # The Lyapunov function is V(x) = ϕ(x) − ϕ(x̂) + λ |R*(x̅−x̅*|₁
        self.xbar_indices = None
        self.xhat_indices = None

        # If set to true, then we only add the states that violate the
        # Lyapunov conditions to the training set.
        self.add_adversarial_state_only = True

    def sample_loss(self, positivity_state_samples, derivative_state_samples,
                    derivative_state_samples_next,
                    lyapunov_positivity_sample_cost_weight,
                    lyapunov_derivative_sample_cost_weight):
        """
        Compute the cost as the summation of
        1. hinge(-V(xⁱ) + ε₂ |xⁱ - x*|₁) for sampled state xⁱ.
        2. hinge(dV(xⁱ) + ε V(xⁱ)) for sampled state xⁱ.
        """
        assert (isinstance(positivity_state_samples, torch.Tensor))
        assert (isinstance(derivative_state_samples, torch.Tensor))
        assert (isinstance(derivative_state_samples_next, torch.Tensor))
        assert (positivity_state_samples.shape[1] ==
                self.lyapunov_hybrid_system.system.x_dim)
        assert (derivative_state_samples.shape[1] ==
                self.lyapunov_hybrid_system.system.x_dim)
        assert (derivative_state_samples_next.shape == (
            derivative_state_samples.shape[0],
            self.lyapunov_hybrid_system.system.x_dim))
        dtype = self.lyapunov_hybrid_system.system.dtype
        if lyapunov_positivity_sample_cost_weight != 0 and\
                positivity_state_samples.shape[0] > 0:
            positivity_sample_loss = lyapunov_positivity_sample_cost_weight *\
                self.lyapunov_hybrid_system.\
                lyapunov_positivity_loss_at_samples(
                    self.x_equilibrium,
                    positivity_state_samples, self.V_lambda,
                    self.lyapunov_positivity_epsilon, R=self.R_options.R(),
                    margin=self.lyapunov_positivity_sample_margin,
                    xbar_indices=self.xbar_indices,
                    xhat_indices=self.xhat_indices)
        else:
            positivity_sample_loss = torch.tensor(0., dtype=dtype)
        if lyapunov_derivative_sample_cost_weight != 0 and\
                derivative_state_samples.shape[0] > 0:
            derivative_sample_loss = lyapunov_derivative_sample_cost_weight *\
                self.lyapunov_hybrid_system.\
                lyapunov_derivative_loss_at_samples_and_next_states(
                    self.V_lambda, self.lyapunov_derivative_epsilon,
                    derivative_state_samples,
                    derivative_state_samples_next, self.x_equilibrium,
                    self.lyapunov_derivative_eps_type, R=self.R_options.R(),
                    margin=self.lyapunov_derivative_sample_margin,
                    xbar_indices=self.xbar_indices,
                    xhat_indices=self.xhat_indices)
        else:
            derivative_sample_loss = torch.tensor(0., dtype=dtype)

        return positivity_sample_loss, derivative_sample_loss

    def solve_positivity_mip(self):
        dtype = self.lyapunov_hybrid_system.system.dtype
        lyapunov_positivity_as_milp_return = self.lyapunov_hybrid_system.\
            lyapunov_positivity_as_milp(
                self.x_equilibrium, self.V_lambda,
                self.lyapunov_positivity_epsilon, R=self.R_options.R(),
                fixed_R=self.R_options.fixed_R,
                x_warmstart=self.lyapunov_positivity_last_x_adv,
                xbar_indices=self.xbar_indices,
                xhat_indices=self.xhat_indices)
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
            self.lyapunov_positivity_last_x_adv = torch.tensor(
                [v.x for v in lyapunov_positivity_as_milp_return[1]],
                dtype=dtype)
        # Now get all the solution as adversarial states.
        positivity_mip_adversarial = []
        for solution_number in range(
                np.min((self.lyapunov_positivity_mip_pool_solutions,
                        lyapunov_positivity_mip.gurobi_model.solCount))):
            lyapunov_positivity_mip.gurobi_model.setParam(
                gurobipy.GRB.Param.SolutionNumber, solution_number)
            if not self.add_adversarial_state_only or \
                (self.add_adversarial_state_only and
                 lyapunov_positivity_mip.gurobi_model.PoolObjVal > 0):
                positivity_mip_adversarial.append(
                    [v.xn for v in lyapunov_positivity_as_milp_return[1]])
        if (len(positivity_mip_adversarial) > 0):
            positivity_mip_adversarial = torch.tensor(
                positivity_mip_adversarial, dtype=dtype)
        else:
            positivity_mip_adversarial = torch.empty(
                (0, self.lyapunov_hybrid_system.system.x_dim), dtype=dtype)
        return lyapunov_positivity_mip, lyapunov_positivity_mip_obj,\
            positivity_mip_adversarial

    def solve_derivative_mip(self):
        dtype = self.lyapunov_hybrid_system.system.dtype
        lyapunov_derivative_as_milp_return = self.lyapunov_hybrid_system.\
            lyapunov_derivative_as_milp(
                self.x_equilibrium, self.V_lambda,
                self.lyapunov_derivative_epsilon,
                self.lyapunov_derivative_eps_type, R=self.R_options.R(),
                fixed_R=self.R_options.fixed_R,
                x_warmstart=self.lyapunov_derivative_last_x_adv,
                xbar_indices=self.xbar_indices,
                xhat_indices=self.xhat_indices)
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

        relu_zeta_val = np.array(
            [np.round(v.x) for v in lyapunov_derivative_as_milp_return[2]])
        if self.output_flag:
            print("lyapunov derivative MIP Relu activation: "
                  f"{np.argwhere(relu_zeta_val == 1).squeeze()}")
            print("adversarial x " +
                  f"{[v.x for v in lyapunov_derivative_as_milp_return[1]]}")
        if self.lyapunov_derivative_mip_warmstart:
            self.lyapunov_derivative_last_x_adv = torch.tensor(
                [v.x for v in lyapunov_derivative_as_milp_return[1]],
                dtype=dtype)
        # Return the solution of the MILP as adversarial states.
        derivative_mip_adversarial = []
        if (isinstance(self.lyapunov_hybrid_system.system,
                       hybrid_linear_system.AutonomousHybridLinearSystem)):
            derivative_mip_adversarial_next = []
        for solution_number in range(
                np.min((self.lyapunov_derivative_mip_pool_solutions,
                        lyapunov_derivative_mip.gurobi_model.solCount))):
            lyapunov_derivative_mip.gurobi_model.setParam(
                gurobipy.GRB.Param.SolutionNumber, solution_number)
            if not self.add_adversarial_state_only or (
                    self.add_adversarial_state_only
                    and lyapunov_derivative_mip.gurobi_model.PoolObjVal > 0):
                derivative_mip_adversarial.append(
                    [v.xn for v in lyapunov_derivative_as_milp_return[1]])

                if (isinstance(
                        self.lyapunov_hybrid_system.system,
                        hybrid_linear_system.AutonomousHybridLinearSystem)):
                    derivative_mip_adversarial_mode = np.argwhere(
                        np.array([
                            v.xn for v in lyapunov_derivative_as_milp_return[3]
                        ]) > 0.99)[0][0]
                    derivative_mip_adversarial_next.append(
                        self.lyapunov_hybrid_system.system.step_forward(
                            torch.tensor(derivative_mip_adversarial[-1],
                                         dtype=dtype),
                            derivative_mip_adversarial_mode))

        if len(derivative_mip_adversarial) > 0:
            derivative_mip_adversarial = torch.tensor(
                derivative_mip_adversarial, dtype=dtype)
            if (isinstance(self.lyapunov_hybrid_system.system,
                           hybrid_linear_system.AutonomousHybridLinearSystem)):
                derivative_mip_adversarial_next = torch.stack(
                    derivative_mip_adversarial_next)
            else:
                derivative_mip_adversarial_next = \
                    self.lyapunov_hybrid_system.system.step_forward(
                        derivative_mip_adversarial)
        else:
            derivative_mip_adversarial = torch.empty(
                (0, self.lyapunov_hybrid_system.system.x_dim), dtype=dtype)
            derivative_mip_adversarial_next = torch.empty(
                (0, self.lyapunov_hybrid_system.system.x_dim), dtype=dtype)

        return lyapunov_derivative_mip, lyapunov_derivative_mip_obj,\
            derivative_mip_adversarial, derivative_mip_adversarial_next

    def total_loss(self, positivity_state_samples, derivative_state_samples,
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
        dtype = self.lyapunov_hybrid_system.system.dtype
        if lyapunov_positivity_mip_cost_weight is not None:
            lyapunov_positivity_mip, lyapunov_positivity_mip_obj,\
                positivity_mip_adversarial = self.solve_positivity_mip()
        else:
            lyapunov_positivity_mip = None
            lyapunov_positivity_mip_obj = None
            positivity_mip_adversarial = None
        if lyapunov_derivative_mip_cost_weight is not None:
            lyapunov_derivative_mip, lyapunov_derivative_mip_obj,\
                derivative_mip_adversarial, derivative_mip_adversarial_next =\
                self.solve_derivative_mip()
        else:
            lyapunov_derivative_mip = None
            lyapunov_derivative_mip_obj = None
            derivative_mip_adversarial = None
            derivative_mip_adversarial_next = None

        loss = torch.tensor(0., dtype=dtype)

        positivity_mip_loss = 0.
        if lyapunov_positivity_mip_cost_weight != 0 and\
                lyapunov_positivity_mip_cost_weight is not None:
            positivity_mip_loss = \
                lyapunov_positivity_mip_cost_weight * \
                lyapunov_positivity_mip.\
                compute_objective_from_mip_data_and_solution(
                    solution_number=0, penalty=1e-13)
        derivative_mip_loss = 0
        if lyapunov_derivative_mip_cost_weight != 0\
                and lyapunov_derivative_mip_cost_weight is not None:
            mip_cost = lyapunov_derivative_mip.\
                compute_objective_from_mip_data_and_solution(
                    solution_number=0, penalty=1e-13)
            derivative_mip_loss = \
                lyapunov_derivative_mip_cost_weight * mip_cost

        # We add the most adverisal states of the positivity MIP and derivative
        # MIP to the training set. Note we solve positivity MIP and derivative
        # MIP separately. This is different from adding the most adversarial
        # state of the total loss.
        if self.add_positivity_adversarial_state and \
                lyapunov_positivity_mip_cost_weight is not None:
            positivity_state_samples = torch.cat([
                positivity_state_samples,
                positivity_mip_adversarial.reshape(
                    (-1, self.lyapunov_hybrid_system.system.x_dim))
            ],
                                                 dim=0)
        if self.add_derivative_adversarial_state and \
                lyapunov_derivative_mip_cost_weight is not None:
            derivative_state_samples = torch.cat([
                derivative_state_samples,
                derivative_mip_adversarial.reshape(
                    (-1, self.lyapunov_hybrid_system.system.x_dim))
            ],
                                                 dim=0)
            derivative_state_samples_next = torch.cat([
                derivative_state_samples_next,
                derivative_mip_adversarial_next.reshape(
                    (-1, self.lyapunov_hybrid_system.system.x_dim))
            ],
                                                      dim=0)

        if positivity_state_samples.shape[0] > self.max_sample_pool_size:
            positivity_state_samples_in_pool = \
                positivity_state_samples[-self.max_sample_pool_size:]
        else:
            positivity_state_samples_in_pool = positivity_state_samples
        if derivative_state_samples.shape[0] > self.max_sample_pool_size:
            derivative_state_samples_in_pool = \
                derivative_state_samples[-self.max_sample_pool_size:]
            derivative_state_samples_next_in_pool = \
                derivative_state_samples_next[-self.max_sample_pool_size:]
        else:
            derivative_state_samples_in_pool = derivative_state_samples
            derivative_state_samples_next_in_pool = \
                derivative_state_samples_next
        positivity_sample_loss, derivative_sample_loss = self.sample_loss(
            positivity_state_samples_in_pool, derivative_state_samples_in_pool,
            derivative_state_samples_next_in_pool,
            lyapunov_positivity_sample_cost_weight,
            lyapunov_derivative_sample_cost_weight)

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
                torch.save(self.lyapunov_hybrid_system.lyapunov_relu,
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
                if attr[0] not in ('lyapunov_hybrid_system',
                                   'summary_writer_folder'):
                    print(f"{attr[0]}: {attr[1]}")
                    if self.enable_wandb:
                        wandb.config.update({attr[0]: f"{attr[1]}"})

    def _training_params(self):
        """
        The parameters to be trained.
        """
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
        return training_params

    def train(self, state_samples_all):
        train_start_time = time.time()
        if self.output_flag:
            self.print()
        assert (isinstance(state_samples_all, torch.Tensor))
        assert (state_samples_all.shape[1] ==
                self.lyapunov_hybrid_system.system.x_dim)
        positivity_state_samples = state_samples_all.clone()
        derivative_state_samples = state_samples_all.clone()
        if (state_samples_all.shape[0] > 0):
            derivative_state_samples_next = torch.stack([
                self.lyapunov_hybrid_system.system.step_forward(
                    derivative_state_samples[i])
                for i in range(derivative_state_samples.shape[0])
            ],
                                                        dim=0)
        else:
            derivative_state_samples_next = torch.empty_like(state_samples_all)
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
                    "derivative MIP cost": lyapunov_derivative_mip_cost,
                    "time": time.time() - train_start_time
                })
            if self.summary_writer_folder is not None:
                writer.add_scalar("loss", loss.item(), iter_count)
                writer.add_scalar("positivity MIP cost",
                                  lyapunov_positivity_mip_cost, iter_count)
                writer.add_scalar("derivative MIP cost",
                                  lyapunov_derivative_mip_cost, iter_count)
            if self.output_flag:
                print(f"Iter {iter_count}, loss {loss}, " +
                      "positivity cost " +
                      f"{lyapunov_positivity_mip_cost}, " +
                      "derivative_cost " + f"{lyapunov_derivative_mip_cost}")
            if lyapunov_positivity_mip_cost <=\
                self.lyapunov_positivity_convergence_tol and\
                lyapunov_derivative_mip_cost <= \
                    self.lyapunov_derivative_convergence_tol:
                return (True, loss.item(), lyapunov_positivity_mip_cost,
                        lyapunov_derivative_mip_cost)
            loss.backward()
            optimizer.step()
            iter_count += 1
        return (False, loss.item(), lyapunov_positivity_mip_cost,
                lyapunov_derivative_mip_cost)

    def train_lyapunov_on_samples(self, state_samples_all, num_epochs,
                                  batch_size):
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
        """
        assert (isinstance(state_samples_all, torch.Tensor))
        assert (state_samples_all.shape[1] ==
                self.lyapunov_hybrid_system.system.x_dim)
        best_loss = np.inf
        training_params = self._training_params()
        optimizer = torch.optim.Adam(training_params, lr=self.learning_rate)
        dataset = torch.utils.data.TensorDataset(state_samples_all)
        train_set_size = int(len(dataset) * 0.8)
        test_set_size = len(dataset) - train_set_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_set_size, test_set_size])
        data_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)
        test_state_samples = test_dataset[:][0]
        for epoch in range(num_epochs):
            running_loss = 0.
            for _, batch_data in enumerate(data_loader):
                state_samples_batch = batch_data[0]
                optimizer.zero_grad()
                state_samples_next = torch.stack([
                    self.lyapunov_hybrid_system.system.step_forward(
                        state_samples_batch[i])
                    for i in range(state_samples_batch.shape[0])
                ],
                                                 dim=0)
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
                    test_state_samples[i])
                for i in range(test_state_samples.shape[0])
            ],
                                                  dim=0)
            test_loss, _, _, _, _, _, _, _, _, _ = self.total_loss(
                test_state_samples,
                test_state_samples,
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

    class AdversarialTrainingOptions:
        def __init__(self):
            # Number of batches per epoch.
            self.num_batches = 10
            # Number of epochs on the adversarial samples before solving an MIP
            # to add new adversarial samples.
            self.num_epochs_per_mip = 20
            # We keep a pool of adversarial samples.
            self.positivity_samples_pool_size = 10000
            self.derivative_samples_pool_size = 10000

        def wandb_config(self):
            for attr in inspect.getmembers(self):
                if not attr[0].startswith('_') and not inspect.ismethod(
                        attr[1]):
                    wandb.config.update({attr[0]: f"{attr[1]}"})

    def _batch_descent_on_samples(self, positivity_state_samples_all,
                                  derivative_state_samples_all, optimizer,
                                  options: AdversarialTrainingOptions):
        """
        Give the samples, divide the samples to small batches, and run several
        epochs to reduce the loss on the sampled states.
        """
        positivity_dataset = torch.utils.data.TensorDataset(
            positivity_state_samples_all)
        derivative_dataset = torch.utils.data.TensorDataset(
            derivative_state_samples_all)
        # TODO(hongkai.dai): currently by using batch_size, I don't guarantee
        # to get options.num_batches batches in the dataset. Write a customized
        # loader later.
        positivity_loader = torch.utils.data.DataLoader(
            positivity_dataset,
            batch_size=int(
                np.ceil(len(positivity_dataset) / options.num_batches)),
            shuffle=True)
        derivative_loader = torch.utils.data.DataLoader(
            derivative_dataset,
            batch_size=int(
                np.ceil(len(derivative_dataset) / options.num_batches)),
            shuffle=True)
        for epoch in range(options.num_epochs_per_mip):
            it_positivity_samples = iter(positivity_loader)
            it_derivative_samples = iter(derivative_loader)
            for i in range(
                    np.min((len(positivity_loader), len(derivative_loader)))):
                optimizer.zero_grad()
                positivity_state_batch = next(it_positivity_samples)[0]
                derivative_state_batch = next(it_derivative_samples)[0]
                derivative_state_next_batch = \
                    self.lyapunov_hybrid_system.system.step_forward(
                        derivative_state_batch)
                positivity_sample_loss, derivative_sample_loss = \
                    self.sample_loss(
                        positivity_state_batch, derivative_state_batch,
                        derivative_state_next_batch,
                        self.lyapunov_positivity_sample_cost_weight,
                        self.lyapunov_derivative_sample_cost_weight)
                batch_loss = positivity_sample_loss +\
                    derivative_sample_loss
                batch_loss.backward()
                optimizer.step()

            derivative_state_samples_next_all =\
                self.lyapunov_hybrid_system.system.step_forward(
                    derivative_state_samples_all)
            positivity_sample_epoch_loss, derivative_sample_epoch_loss = \
                self.sample_loss(
                    positivity_state_samples_all,
                    derivative_state_samples_all,
                    derivative_state_samples_next_all,
                    self.lyapunov_positivity_sample_cost_weight,
                    self.lyapunov_derivative_sample_cost_weight)
            if self.output_flag:
                print(f"epoch {epoch}, positivity_sample_loss " +
                      f"{positivity_sample_epoch_loss.item()}, " +
                      "derivative_sample_loss " +
                      f"{derivative_sample_epoch_loss.item()}")
            if positivity_sample_epoch_loss == 0. and\
                    derivative_sample_epoch_loss == 0.:
                return

    def train_adversarial(self, positivity_state_samples_init: torch.Tensor,
                          derivative_state_samples_init: torch.Tensor,
                          options: AdversarialTrainingOptions):
        """
        We solve the MILP as verifier. If the MILP finds counter-examples, we
        add the counter-examples to the training set (with a maximal buffer
        size), and do gradient descent on this training set.
        @param positivity_state_samples_init The initial training set for the
        Lyapunov positivity condition.
        @param derivative_state_samples_init The initial training set for the
        derivative condition.
        """
        assert (self.add_derivative_adversarial_state)
        assert (self.add_positivity_adversarial_state)
        if self.output_flag:
            self.print()
        if self.enable_wandb:
            options.wandb_config()

        train_start_time = time.time()
        positivity_state_samples_all = positivity_state_samples_init.clone()
        derivative_state_samples_all = derivative_state_samples_init.clone()
        training_params = self._training_params()
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(training_params,
                                         lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(training_params,
                                        lr=self.learning_rate,
                                        momentum=self.momentum)
        iter_count = 0
        while iter_count < self.max_iterations:
            # Now solve MIP to find adversarial states.
            lyapunov_positivity_mip, lyapunov_positivity_mip_obj,\
                positivity_mip_adversarial = self.solve_positivity_mip()
            lyapunov_derivative_mip, lyapunov_derivative_mip_obj,\
                derivative_mip_adversarial, _ = self.solve_derivative_mip()
            positivity_state_samples_all = torch.cat(
                (positivity_state_samples_all, positivity_mip_adversarial),
                dim=0)
            derivative_state_samples_all = torch.cat(
                (derivative_state_samples_all, derivative_mip_adversarial),
                dim=0)
            if positivity_state_samples_all.shape[
                    0] > options.positivity_samples_pool_size:
                positivity_state_samples_all = positivity_state_samples_all[
                    -options.positivity_samples_pool_size:, :]
            if derivative_state_samples_all.shape[
                    0] > options.derivative_samples_pool_size:
                derivative_state_samples_all = derivative_state_samples_all[
                    -options.derivative_samples_pool_size:, :]
            if self.output_flag:
                print(f"Iter {iter_count}, positivity cost " +
                      f"{lyapunov_positivity_mip_obj}, " + "derivative_cost " +
                      f"{lyapunov_derivative_mip_obj}")
            if self.enable_wandb:
                wandb.log({
                    "positivity MIP cost": lyapunov_positivity_mip_obj,
                    "derivative MIP cost": lyapunov_derivative_mip_obj,
                    "time": time.time() - train_start_time
                })
            if lyapunov_positivity_mip_obj < \
                self.lyapunov_positivity_convergence_tol and\
                lyapunov_derivative_mip_obj < \
                    self.lyapunov_derivative_convergence_tol:
                return True, positivity_state_samples_all,\
                    derivative_state_samples_all
            # Now do gradient descent on the adversarial states.
            self._batch_descent_on_samples(positivity_state_samples_all,
                                           derivative_state_samples_all,
                                           optimizer, options)
            iter_count += 1
        return False, positivity_state_samples_all,\
            derivative_state_samples_all


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

    def train_with_cost_to_go(self, network, x0_value_samples, V_lambda,
                              x_equilibrium, R):
        """
        Similar to train() function, but with given samples on initial_state
        and cost-to-go.
        """
        if R is None:
            x_dim = x_equilibrium.numel()
            R = torch.eye(x_dim, dtype=torch.float64)
        state_samples_all = torch.stack([pair[0] for pair in x0_value_samples],
                                        dim=0)
        value_samples_all = torch.stack([pair[1] for pair in x0_value_samples])
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=self.learning_rate)
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

    def train(self,
              system,
              network,
              V_lambda,
              x_equilibrium,
              instantaneous_cost_fun,
              x0_samples,
              T,
              discrete_time_flag,
              R,
              x_goal=None,
              pruner=None):
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
        assert (isinstance(system,
                           hybrid_linear_system.AutonomousHybridLinearSystem))
        assert (isinstance(x_equilibrium, torch.Tensor))
        assert (x_equilibrium.shape == (system.x_dim, ))
        assert (isinstance(V_lambda, float))
        assert (isinstance(x0_samples, torch.Tensor))
        x0_value_samples = hybrid_linear_system.generate_cost_to_go_samples(
            system, [x0_samples[i] for i in range(x0_samples.shape[0])], T,
            instantaneous_cost_fun, discrete_time_flag, x_goal, pruner)
        return self.train_with_cost_to_go(network, x0_value_samples, V_lambda,
                                          x_equilibrium, R)
