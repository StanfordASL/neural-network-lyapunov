import torch
import gurobipy


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
        lyapunov_positivity_loss_at_sample() (which represents hinge(-V(xⁱ)))
        lyapunov_derivative_as_milp() (which represents maxₓ dV(x) + ε V(x))
        lyapunov_derivative_loss_at_sample() (which represents
        hinge(dV(xⁱ) + ε V(xⁱ)).
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
        # The number of samples xⁱ in each batch.
        self.batch_size = 100
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
        self.lyapunov_derivative_sample_margin = 0.1
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

    def total_loss(self, relu, state_samples_all, state_samples_next):
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
        assert(isinstance(state_samples_all, list))
        assert(isinstance(state_samples_next, list))
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

        loss = torch.tensor(0., dtype=dtype)
        state_sample_indices = torch.randint(
            0, len(state_samples_all), (self.batch_size,))
        relu_at_equilibrium = relu.forward(self.x_equilibrium)
        for i in state_sample_indices:
            for state_sample_next_i in state_samples_next[i]:
                loss += self.lyapunov_derivative_sample_cost_weight *\
                    self.lyapunov_hybrid_system.\
                    lyapunov_derivative_loss_at_sample_and_next_state(
                        relu, self.V_rho, self.lyapunov_derivative_epsilon,
                        state_samples_all[i], state_sample_next_i,
                        self.x_equilibrium,
                        margin=self.lyapunov_derivative_sample_margin)
            loss += self.lyapunov_positivity_sample_cost_weight *\
                self.lyapunov_hybrid_system.lyapunov_positivity_loss_at_sample(
                    relu, relu_at_equilibrium, self.x_equilibrium,
                    state_samples_all[i], self.V_rho,
                    margin=self.lyapunov_positivity_sample_margin)

        for mip_sol_number in range(
                self.lyapunov_positivity_mip_pool_solutions):
            if mip_sol_number < lyapunov_positivity_mip.gurobi_model.solCount:
                loss += self.lyapunov_positivity_mip_cost_weight * \
                    torch.pow(torch.tensor(
                        self.lyapunov_positivity_mip_cost_decay_rate,
                        dtype=dtype), mip_sol_number) *\
                    -lyapunov_positivity_mip.\
                    compute_objective_from_mip_data_and_solution(
                        solution_number=mip_sol_number, penalty=1e-12)
        for mip_sol_number in range(
                self.lyapunov_derivative_mip_pool_solutions):
            if (mip_sol_number <
                    lyapunov_derivative_mip.gurobi_model.solCount):
                mip_cost = lyapunov_derivative_mip.\
                    compute_objective_from_mip_data_and_solution(
                        solution_number=mip_sol_number, penalty=1e-12)
                loss += self.lyapunov_derivative_mip_cost_weight *\
                    torch.pow(torch.tensor(
                        self.lyapunov_derivative_mip_cost_decay_rate,
                        dtype=dtype), mip_sol_number) * mip_cost
                lyapunov_derivative_mip.gurobi_model.setParam(
                    gurobipy.GRB.Param.SolutionNumber, mip_sol_number)
        return loss, lyapunov_positivity_mip.gurobi_model.ObjVal,\
            lyapunov_derivative_mip.gurobi_model.ObjVal

    def train(self, relu, state_samples_all):
        assert(isinstance(state_samples_all, list))
        state_samples_next = \
            [self.lyapunov_hybrid_system.system.possible_next_states(x) for x
             in state_samples_all]
        iter_count = 0
        optimizer = torch.optim.Adam(
            relu.parameters(), lr=self.learning_rate)
        while iter_count < self.max_iterations:
            optimizer.zero_grad()
            loss, lyapunov_positivity_mip_cost, lyapunov_derivative_mip_cost \
                = self.total_loss(relu, state_samples_all, state_samples_next)
            if self.output_flag:
                print(f"Iter {iter_count}, loss {loss}, " +
                      f"positivity cost {lyapunov_positivity_mip_cost}, " +
                      f"derivative_cost {lyapunov_derivative_mip_cost}")
            if lyapunov_positivity_mip_cost >= 0 and\
                    lyapunov_derivative_mip_cost <= 0:
                return True
            loss.backward()
            optimizer.step()
            iter_count += 1
        return False


class TrainValueApproximatorOptions:
    def __init__(self):
        self.max_epochs = 100
        # When the training error is less than this tolerance, stop the
        # training.
        self.convergence_tolerance = 1e-3
        self.learning_rate = 0.02
        # Number of steps for the cost-to-go horizon.
        self.num_steps = 100


def train_value_approximator(
    hybrid_linear_system, relu, V_rho, x_equilibrium, instantaneous_cost_fun,
        state_samples_all, options):
    """
    Train a ReLU network such that ReLU(x) - ReLU(x*) + ρ|x-x*|₁ approximates
    the cost-to-go (value function) of a hybrid linear system.
    @param hybrid_linear_system This system has to define a cost_to_go(x)
    function, that computes the cost-to-go from a state x.
    @param relu Both an input and an output paramter. As an input, it
    represents the initial guess of the network. As an output, it represents
    the trained network.
    @param V_rho ρ in the documentation above.
    @param x_equilibrium The equilibrium state x*.
    @param instantaneous_cost_fun A function evaluates the instantaneous cost
    of a state.
    @param state_samples_all A list of state samples.
    @param options A TrainValueApproximatorOptions instance.
    @return is_converged Whether the training converged or not.
    """
    assert(isinstance(x_equilibrium, torch.Tensor))
    assert(x_equilibrium.shape == (hybrid_linear_system.x_dim,))
    assert(isinstance(V_rho, float))
    assert(isinstance(state_samples_all, list))
    assert(isinstance(options, TrainValueApproximatorOptions))
    value_samples = [hybrid_linear_system.cost_to_go(
        state, instantaneous_cost_fun, options.num_steps)
        for state in state_samples_all]

    optimizer = torch.optim.Adam(relu.parameters(), lr=options.learning_rate)
    for epoch in range(options.max_epochs):
        optimizer.zero_grad()
        state_samples_all_torch = torch.stack(state_samples_all, dim=0)
        relu_output = relu(state_samples_all_torch)
        relu_x_equilibrium = relu.forward(x_equilibrium)
        value_relu = relu_output.squeeze() - relu_x_equilibrium +\
            V_rho * torch.norm(
                state_samples_all_torch - x_equilibrium.reshape((1, -1)).
                expand(state_samples_all_torch.shape[0], -1), dim=1, p=1)
        loss = torch.nn.MSELoss()(
            value_relu, torch.stack(value_samples, dim=0))
        if (loss.item() <= options.convergence_tolerance):
            return True
        loss.backward()
        optimizer.step()
    return False
