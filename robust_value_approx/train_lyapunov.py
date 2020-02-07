import torch
import numpy as np
import gurobipy
import copy
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.relu_to_optimization as relu_to_optimization


class TrainLyapunovReLU:
    """
    We will train a ReLU network, such that the function
    V(x) = ReLU(x) - ReLU(x*) + ρ|x-x*|₁ is a Lyapunov function that certifies
    (exponential) convergence. Namely V(x) should satisfy the following
    conditions
    1. V(x) > 0 ∀ x ≠ x*
    2. dV(x) ≤ -ε V(x) ∀ x
    where dV(x) = V̇(x) for continuous time system, and
    dV(x[n]) = V(x[n+1]) - V(x[n]) for discrete time system.
    In order to find such V(x), we penalize a (weighted) sum of the following
    loss
    1. hinge(-V(xⁱ)) for sampled state xⁱ.
    2. hinge(dV(xⁱ) + ε V(xⁱ)) for sampled state xⁱ.
    3. -min_x V(x) - ε₂ |x - x*|₁
    4. max_x dV(x) + ε V(x)
    where ε₂ is a given positive scalar, and |x - x*|₁ is the 1-norm of x - x*.
    hinge(z) = max(z + margin, 0) where margin is a given scalar.
    """

    def __init__(self, lyapunov_hybrid_system, V_rho, x_equilibrium):
        """
        @param lyapunov_hybrid_system This input should define a common
        interface
        lyapunov_positivity_as_milp() (which represents
        minₓ V(x) - ε₂ |x - x*|₁)
        lyapunov_positivity_loss_at_samples() (which represents
        mean(hinge(-V(xⁱ))))
        lyapunov_derivative_as_milp() (which represents maxₓ dV(x) + ε V(x))
        lyapunov_derivative_loss_at_samples() (which represents
        mean(hinge(dV(xⁱ) + ε V(xⁱ))).
        One example of input type is lyapunov.LyapunovDiscreteTimeHybridSystem.
        @param V_rho ρ in the documentation above.
        @param x_equilibrium The equilibrium state.
        """
        self.lyapunov_hybrid_system = lyapunov_hybrid_system
        assert(isinstance(V_rho, float))
        self.V_rho = V_rho
        assert(isinstance(x_equilibrium, torch.Tensor))
        assert(x_equilibrium.shape == (lyapunov_hybrid_system.system.x_dim,))
        self.x_equilibrium = x_equilibrium
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
        self.lyapunov_positivity_mip_pool_solutions = 10
        # The weight of max_x dV(x) + εV(x)
        self.lyapunov_derivative_mip_cost_weight = 10.
        # The number of (sub)optimal solutions for the MIP
        # max_x dV(x) + εV(x)
        self.lyapunov_derivative_mip_pool_solutions = 10
        # If set to true, we will print some messages during training.
        self.output_flag = False
        # The MIP objective loss is ∑ⱼ rʲ * j_th_objective. r is the decay
        # rate.
        self.lyapunov_positivity_mip_cost_decay_rate = 0.9
        self.lyapunov_derivative_mip_cost_decay_rate = 0.9
        # This is ε₂ in  V(x) >=  ε₂ |x - x*|₁
        self.lyapunov_positivity_epsilon = 0.01
        # This is ε in dV(x) ≤ -ε V(x)
        self.lyapunov_derivative_epsilon = 0.01

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

    def total_loss(
            self, relu, state_samples_all, state_samples_next,
            lyapunov_positivity_sample_cost_weight=None,
            lyapunov_derivative_sample_cost_weight=None,
            lyapunov_positivity_mip_cost_weight=None,
            lyapunov_derivative_mip_cost_weight=None):
        """
        Compute the total loss as the summation of
        1. hinge(-V(xⁱ)) for sampled state xⁱ.
        2. hinge(dV(xⁱ) + ε V(xⁱ)) for sampled state xⁱ.
        3. -min_x V(x) - ε₂ |x - x*|₁
        4. max_x dV(x) + ε V(x)
        @param relu The ReLU network.
        @param state_samples_all A list. state_samples_all[i] is the i'th
        sampled state.
        @param state_samples_next. The next state(s) of the sampled state.
        state_samples_next[i] is a list containing all the possible next
        state(s) of state_samples_all[i].
        @param loss, positivity_mip_objective, derivative_mip_objective
        positivity_mip_objective is the objective value
        min_x V(x) - ε₂ |x - x*|₁. We want this value to be non-negative.
        derivative_mip_objective is the objective value max_x dV(x) + ε V(x).
        We want this value to be non-positive.
        """
        assert(isinstance(state_samples_all, torch.Tensor))
        assert(isinstance(state_samples_next, torch.Tensor))
        assert(
            state_samples_all.shape[1] ==
            self.lyapunov_hybrid_system.system.x_dim)
        assert(
            state_samples_next.shape[1] ==
            self.lyapunov_hybrid_system.system.x_dim)
        dtype = self.lyapunov_hybrid_system.system.dtype
        lyapunov_positivity_as_milp_return = self.lyapunov_hybrid_system.\
            lyapunov_positivity_as_milp(
                relu, self.x_equilibrium, self.V_rho,
                self.lyapunov_positivity_epsilon)
        lyapunov_positivity_mip = lyapunov_positivity_as_milp_return[0]
        lyapunov_positivity_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        lyapunov_positivity_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSearchMode, 2)
        lyapunov_positivity_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSolutions,
            self.lyapunov_positivity_mip_pool_solutions)
        lyapunov_positivity_mip.gurobi_model.optimize()

        lyapunov_derivative_as_milp_return = self.lyapunov_hybrid_system.\
            lyapunov_derivative_as_milp(
                relu, self.x_equilibrium, self.V_rho,
                self.lyapunov_derivative_epsilon)
        lyapunov_derivative_mip = lyapunov_derivative_as_milp_return[0]
        lyapunov_derivative_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        lyapunov_derivative_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSearchMode, 2)
        lyapunov_derivative_mip.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSolutions,
            self.lyapunov_derivative_mip_pool_solutions)
        lyapunov_derivative_mip.gurobi_model.optimize()
        
        relu_zeta_val = np.array([
            np.round(v.x) for v in lyapunov_derivative_as_milp_return[2]])
        relu_activation_pattern = relu_to_optimization.\
            relu_activation_binary_to_pattern(relu, relu_zeta_val)
        relu_gradient, _, _, _ = relu_to_optimization.\
            ReLUGivenActivationPattern(
                relu, self.lyapunov_hybrid_system.system.x_dim,
                relu_activation_pattern,
                self.lyapunov_hybrid_system.system.dtype)
        print(f"relu gradient {relu_gradient.squeeze().detach().numpy()}")
        print(f"lyapunov derivative MIP Relu activation: "
              f"{np.argwhere(relu_zeta_val == 1).squeeze()}")
        print(f"adversarial x {[v.x for v in lyapunov_derivative_as_milp_return[1]]}")
        print(f"hybrid mode {np.argwhere(np.array([np.round(v.x) == 1 for v in lyapunov_derivative_as_milp_return[3]]))}")

        loss = torch.tensor(0., dtype=dtype)
        relu_at_equilibrium = relu.forward(self.x_equilibrium)
        def set_weight(val, default_val):
            return val if val is not None else default_val
        lyapunov_positivity_sample_cost_weight = set_weight(
            lyapunov_positivity_sample_cost_weight,
            self.lyapunov_positivity_sample_cost_weight)
        if lyapunov_positivity_sample_cost_weight != 0:
            loss += lyapunov_positivity_sample_cost_weight *\
                self.lyapunov_hybrid_system.\
                lyapunov_positivity_loss_at_samples(
                    relu, relu_at_equilibrium, self.x_equilibrium,
                    state_samples_all, self.V_rho,
                    margin=self.lyapunov_positivity_sample_margin)
        lyapunov_derivative_sample_cost_weight = set_weight(
            lyapunov_derivative_sample_cost_weight,
            self.lyapunov_derivative_sample_cost_weight)
        if lyapunov_derivative_sample_cost_weight != 0:
            loss += lyapunov_derivative_sample_cost_weight *\
                self.lyapunov_hybrid_system.\
                lyapunov_derivative_loss_at_samples_and_next_states(
                    relu, self.V_rho, self.lyapunov_derivative_epsilon,
                    state_samples_all, state_samples_next,
                    self.x_equilibrium,
                    margin=self.lyapunov_derivative_sample_margin)

        lyapunov_positivity_mip_cost_weight = set_weight(
            lyapunov_positivity_mip_cost_weight,
            self.lyapunov_positivity_mip_cost_weight)
        if lyapunov_positivity_mip_cost_weight != 0 and np.abs(
            lyapunov_positivity_mip.gurobi_model.ObjVal)>\
                self.lyapunov_positivity_convergence_tol:
            for mip_sol_number in range(
                    self.lyapunov_positivity_mip_pool_solutions):
                if mip_sol_number < \
                        lyapunov_positivity_mip.gurobi_model.solCount:
                    loss += lyapunov_positivity_mip_cost_weight * \
                        torch.pow(torch.tensor(
                            self.lyapunov_positivity_mip_cost_decay_rate,
                            dtype=dtype), mip_sol_number) *\
                        -lyapunov_positivity_mip.\
                        compute_objective_from_mip_data_and_solution(
                            solution_number=mip_sol_number, penalty=1e-13)
        lyapunov_derivative_mip_cost_weight = set_weight(
            lyapunov_derivative_mip_cost_weight,
            self.lyapunov_derivative_mip_cost_weight)
        if lyapunov_derivative_mip_cost_weight != 0:
            for mip_sol_number in range(
                    self.lyapunov_derivative_mip_pool_solutions):
                if (mip_sol_number <
                        lyapunov_derivative_mip.gurobi_model.solCount):
                    mip_cost = lyapunov_derivative_mip.\
                        compute_objective_from_mip_data_and_solution(
                            solution_number=mip_sol_number, penalty=1e-13)
                    loss += lyapunov_derivative_mip_cost_weight *\
                        torch.pow(torch.tensor(
                            self.lyapunov_derivative_mip_cost_decay_rate,
                            dtype=dtype), mip_sol_number) * mip_cost
                    lyapunov_derivative_mip.gurobi_model.setParam(
                        gurobipy.GRB.Param.SolutionNumber, mip_sol_number)
        return loss, lyapunov_positivity_mip.gurobi_model.ObjVal,\
            lyapunov_derivative_mip.gurobi_model.ObjVal

    def train(self, relu, state_samples_all):
        assert(isinstance(state_samples_all, torch.Tensor))
        assert(
            state_samples_all.shape[1] ==
            self.lyapunov_hybrid_system.system.x_dim)
        state_samples_next = torch.stack([
            self.lyapunov_hybrid_system.system.step_forward(
                state_samples_all[i]) for i in
            range(state_samples_all.shape[0])], dim=0)
        iter_count = 0
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                relu.parameters(), lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                relu.parameters(), lr=self.learning_rate, momentum=0)
        else:
            raise Exception(
                "train: unknown optimizer, only support Adam or SGD.")
        lyapunov_positivity_mip_costs = [None] * self.max_iterations
        lyapunov_derivative_mip_costs = [None] * self.max_iterations
        losses = [None] * self.max_iterations
        while iter_count < self.max_iterations:
            optimizer.zero_grad()
            loss, lyapunov_positivity_mip_costs[iter_count],\
                lyapunov_derivative_mip_costs[iter_count] \
                = self.total_loss(relu, state_samples_all, state_samples_next)
            losses[iter_count] = loss.item()
            if self.output_flag:
                print(f"Iter {iter_count}, loss {loss}, " +
                      f"positivity cost " +
                      f"{lyapunov_positivity_mip_costs[iter_count]}, " +
                      f"derivative_cost " +
                      f"{lyapunov_derivative_mip_costs[iter_count]}")
            if lyapunov_positivity_mip_costs[iter_count] >=\
                -self.lyapunov_positivity_convergence_tol and\
                lyapunov_derivative_mip_costs[iter_count] <= \
                    self.lyapunov_derivative_convergence_tol:
                return (True, losses[:iter_count+1],
                        lyapunov_positivity_mip_costs[:iter_count+1],
                        lyapunov_derivative_mip_costs[:iter_count+1])
            loss.backward()
            optimizer.step()
            iter_count += 1
        return (False, losses, lyapunov_positivity_mip_costs,
                lyapunov_derivative_mip_costs)

    def train_lyapunov_on_samples(
        self, relu, state_samples_all, max_iterations,
            max_increasing_iterations):
        """
        Train a ReLU network on given state samples (not the adversarial states
        found by MIP). The loss function is the weighted sum of the lyapunov
        positivity condition violation and the lyapunov derivative condition
        violation on these samples. We stop the training when either the
        maximum iteration is reached, or when many consecutive iterations the
        MIP costs keeps increasing (which means the network overfits to the
        training data). Return the best network (the one with the minimal
        MIP loss) found so far.
        @param relu The neural network
        @param state_samples_all A torch tensor, state_samples_all[i] is the
        i'th sample
        @param max_iterations The maximal number of iterations.
        @param max_increasing_iterations If the MIP loss keeps increasing for
        this number of iterations, stop the training.
        """
        assert(isinstance(state_samples_all, torch.Tensor))
        assert(
            state_samples_all.shape[1] ==
            self.lyapunov_hybrid_system.system.x_dim)
        state_samples_next = torch.stack([
            self.lyapunov_hybrid_system.system.step_forward(
                state_samples_all[i]) for i in
            range(state_samples_all.shape[0])], dim=0)
        best_loss = np.inf
        num_increasing_iterations = 0
        optimizer = torch.optim.Adam(
            relu.parameters(), lr=self.learning_rate)
        previous_mip_loss = np.inf 
        for iter_count in range(max_iterations):
            optimizer.zero_grad()
            loss, lyapunov_positivity_mip_cost,lyapunov_derivative_mip_cost \
                = self.total_loss(
                    relu, state_samples_all, state_samples_next,
                    lyapunov_positivity_mip_cost_weight=0.,
                    lyapunov_derivative_mip_cost_weight=0.)
            mip_loss = self.lyapunov_positivity_mip_cost_weight * \
                -lyapunov_positivity_mip_cost + \
                self.lyapunov_derivative_mip_cost_weight * \
                lyapunov_derivative_mip_cost
            if mip_loss < best_loss:
                best_loss = mip_loss
                best_relu = copy.deepcopy(relu)
            if mip_loss > previous_mip_loss:
                num_increasing_iterations += 1
                if num_increasing_iterations == max_increasing_iterations:
                    print(f"mip loss {mip_loss}")
                    return best_relu
            else:
                # reset the counter.
                num_increasing_iterations = 0
            previous_mip_loss = mip_loss
            if self.output_flag:
                print(f"Iter {iter_count}, loss {loss}, " +
                      f"positivity cost " +
                      f"{lyapunov_positivity_mip_cost}, " +
                      f"derivative_cost " +
                      f"{lyapunov_derivative_mip_cost}, " + 
                      f"mip loss {mip_loss}\n")
            loss.backward()
            optimizer.step()
        print(f"mip loss {mip_loss}")
        return best_relu


class TrainValueApproximator:
    """
    Given a piecewise affine system and some sampled initial state, compute the
    cost-to-go for these sampled states, and then train a network to
    approximate the cost-to-go, such that
    network(x) - network(x*) + ρ |x-x*|₁ ≈ cost_to_go(x)
    """
    def __init__(self):
        self.max_epochs = 100
        self.convergence_tolerance = 1e-3
        self.learning_rate = 0.02

    def train_with_cost_to_go(
            self, network, x0_value_samples, V_rho, x_equilibrium):
        """
        Similar to train() function, but with given samples on initial_state
        and cost-to-go.
        """
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
                V_rho * torch.norm(
                    state_samples_all - x_equilibrium.reshape((1, -1)).
                    expand(state_samples_all.shape[0], -1), dim=1, p=1)
            loss = torch.nn.MSELoss()(value_relu, value_samples_all)
            if (loss.item() <= self.convergence_tolerance):
                return True, loss.item()
            loss.backward()
            optimizer.step()
        return False, loss.item()

    def train(
        self, system, network, V_rho, x_equilibrium, instantaneous_cost_fun,
            x0_samples, T, discrete_time_flag, x_goal=None, pruner=None):
        """
        Train a network such that
        network(x) - network(x*) + ρ*|x-x*|₁ ≈ cost_to_go(x)
        @param system An AutonomousHybridLinearSystem instance.
        @param network a pytorch neural network.
        @param V_rho ρ.
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
        assert(isinstance(V_rho, float))
        assert(isinstance(x0_samples, torch.Tensor))
        x0_value_samples = hybrid_linear_system.generate_cost_to_go_samples(
            system, [x0_samples[i] for i in range(x0_samples.shape[0])], T,
            instantaneous_cost_fun, discrete_time_flag, x_goal, pruner)
        return self.train_with_cost_to_go(
            network, x0_value_samples, V_rho, x_equilibrium)
