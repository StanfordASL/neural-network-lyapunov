import robust_value_approx.lyapunov as lyapunov
import torch
import gurobipy


class LyapunovReluTrainingOptions:
    def __init__(self):
        # The number of samples xⁱ in each batch.
        self.batch_size = 100
        # The learning rate of the optimizer
        self.learning_rate = 0.003
        # The weights of the MILP cost maxₓ c(x) in the total loss.
        self.mip_cost_weights = 10
        # The number of (sub)optimal solutions of the MILP cost. We will sum
        # over the first mip_pool_solutions as the cost maxₓ c(x)
        self.mip_pool_solutions = 10
        # Number of iterations in the training.
        self.max_iterations = 1000
        # If set to true, we will print some message during training.
        self.output_flag = False
        # The MIP objective loss is ∑ⱼ rʲ * j_th_objective. r is
        # mip_cost_decay_rate.
        self.mip_cost_decay_rate = 0.9
        # The loss for the i'th sample is max(-c(xⁱ) + margin, 0). If we want
        # the Lyapunov derivative to be strictly negative, then set this margin
        # to a positive number.
        self.sample_lyapunov_loss_margin = 0.
        # A strictly positive number. For all x[n] outside of the sublevel set
        # V(x) <= ρ, we want V(x[n+1]) - V(x[n]) to be <= -dV_margin.
        self.dV_margin = 0.1
        # A strictly positive number. We want to prove that all states converge
        # to the sublevel set V(x) <= ρ, where ρ is lyapunov_sublevel.
        self.lyapunov_sublevel = 0.1
        # We will prove the exponential convergence rate dV <= -epsilon * V
        self.dV_epsilon = 0.01


def train_lyapunov_relu(
        lyapunov_hybrid_system, relu, x_equilibrium, state_samples_all,
        options):
    """
    We will train a ReLU network to represent the (control) Lyapunov function
    for piecewise affine systems (in either discrete or continuous time). The
    Lyapunov condition can be written as c(x) ≤ 0 ∀x, where c(x) is
    V(x[n+1]) - V(x[n]) for discrete time system, or ∂V/∂x * ẋ for continuous
    time system. To do so, we define the loss function as
    min ∑ᵢ hinge_loss(c(xⁱ)) + weight * maxₓ c(x)
    where hinge_loss is max(-x + margin, 0), xⁱ is the i'th sample of the
    state.
    The training stops at either the iteration limit is achieved, or when the
    following condition are satisfied
    1. maxₓ c(x) ≤ 0 ∀x
    2. c(x) < -dV_margin ∀x s.t V(x) > ρ
    @param lyapunov_hybrid_system This input should define a common interface
    lyapunov_gradient_as_milp() (which represents the MIP maxₓ c(x)) and
    lyapunov_gradient_loss_at_sample() (which represents hinge_loss(c(xⁱ)).
    One example of input type is lyapunov.LyapunovDiscreteTimeHybridSystem.
    @param relu Both as an input and an output. As an input, this represents
    the initial guess of the ReLU network. As an output, this represents the
    network after training.
    @param x_equilibrium The equilibrium state.
    @param state_samples_all A list of torch 1D tensors. Each torch 1D tensor
    is a sampled state xⁱ.
    @param options A LyapunovReluTrainingOptions object.
    @return lyapunov_condition_satisfied At termination, whether the Lyapunov
    condition is satisfied or not.
    """
    assert(isinstance(
        lyapunov_hybrid_system, lyapunov.LyapunovDiscreteTimeHybridSystem))
    assert(isinstance(state_samples_all, list))
    assert(isinstance(x_equilibrium, torch.Tensor))
    assert(isinstance(options, LyapunovReluTrainingOptions))

    dtype = lyapunov_hybrid_system.system.dtype
    iter_count = 0
    state_samples_next = [lyapunov_hybrid_system.system.step_forward(x) for
                          x in state_samples_all]
    while iter_count < options.max_iterations:
        lyapunov_gradient_as_milp_return = lyapunov_hybrid_system.\
            lyapunov_gradient_as_milp(relu, x_equilibrium, options.dV_epsilon)
        mip = lyapunov_gradient_as_milp_return[0]
        mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        mip.gurobi_model.setParam(gurobipy.GRB.Param.PoolSearchMode, 2)
        mip.gurobi_model.setParam(gurobipy.GRB.Param.PoolSolutions,
                                  options.mip_pool_solutions)
        mip.gurobi_model.optimize()

        sublevelset_as_milp_return = lyapunov_hybrid_system.\
            lyapunov_gradient_as_milp(
                relu, x_equilibrium, options.dV_epsilon,
                options.lyapunov_sublevel)
        mip_sublevel = sublevelset_as_milp_return[0]
        mip_sublevel.gurobi_model.setParam(
            gurobipy.GRB.Param.OutputFlag, False)
        mip_sublevel.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSearchMode, 2)
        mip_sublevel.gurobi_model.setParam(
            gurobipy.GRB.Param.PoolSolutions, options.mip_pool_solutions)
        mip_sublevel.gurobi_model.optimize()

        if (options.output_flag):
            print("Iteration: {}, MIP objective:{}, Sublevel MIP obj:{}".
                  format(iter_count, mip.gurobi_model.ObjVal,
                         mip_sublevel.gurobi_model.ObjVal))
        if (mip.gurobi_model.ObjVal <= 0. and
                mip_sublevel.gurobi_model.ObjVal <= -options.dV_margin):
            return True

        loss = torch.tensor(0., dtype=dtype)
        state_sample_indices = torch.randint(
            0, len(state_samples_all), (options.batch_size,))
        for i in state_sample_indices:
            loss += lyapunov_hybrid_system.\
                lyapunov_loss_at_sample_and_next_state(
                    relu, state_samples_all[i], state_samples_next[i],
                    margin=options.sample_lyapunov_loss_margin)

        for mip_sol_number in range(options.mip_pool_solutions):
            if (mip_sol_number < mip.gurobi_model.solCount):
                loss += torch.pow(
                    torch.tensor(options.mip_cost_decay_rate, dtype=dtype),
                    mip_sol_number) *\
                        mip.compute_objective_from_mip_data_and_solution(
                            solution_number=mip_sol_number, penalty=1e-8)
        for mip_sol_number in range(options.mip_pool_solutions):
            if (mip_sol_number < mip_sublevel.gurobi_model.solCount):
                loss += torch.pow(
                    torch.tensor(options.mip_cost_decay_rate, dtype=dtype),
                    mip_sol_number) *\
                        torch.nn.HingeEmbeddingLoss(margin=options.dV_margin)(
                            -mip_sublevel.
                            compute_objective_from_mip_data_and_solution(
                                solution_number=mip_sol_number),
                            torch.tensor(-1.))

        if (options.output_flag):
            print("Loss:{}".format(loss))
        optimizer = torch.optim.Adam(relu.parameters(),
                                     lr=options.learning_rate)
        optimizer.zero_grad()
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
    hybrid_linear_system, relu, instantaneous_cost_fun, state_samples_all,
        options):
    """
    Train a ReLU network to approximates the cost-to-go (value function) of a
    hybrid linear system.
    @param hybrid_linear_system This system has to define a cost_to_go(x)
    function, that computes the cost-to-go from a state x.
    @param relu Both an input and an output paramter. As an input, it
    represents the initial guess of the network. As an output, it represents
    the trained network.
    @param instantaneous_cost_fun A function evaluates the instantaneous cost
    of a state.
    @param state_samples_all A list of state samples.
    @param options A TrainValueApproximatorOptions instance.
    @return is_converged Whether the training converged or not.
    """
    assert(isinstance(state_samples_all, list))
    assert(isinstance(options, TrainValueApproximatorOptions))
    value_samples = [hybrid_linear_system.cost_to_go(
        state, instantaneous_cost_fun, options.num_steps)
        for state in state_samples_all]

    optimizer = torch.optim.Adam(relu.parameters(), lr=options.learning_rate)
    for epoch in range(options.max_epochs):
        optimizer.zero_grad()
        relu_output = relu(torch.stack(state_samples_all, dim=0))
        loss = torch.nn.MSELoss()(
            relu_output.squeeze(), torch.stack(value_samples, dim=0))
        if (loss.item() <= options.convergence_tolerance):
            return True
        loss.backward()
        optimizer.step()
    return False
