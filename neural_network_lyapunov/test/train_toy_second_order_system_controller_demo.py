import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.relu_system as relu_system

import torch

if __name__ == "__main__":
    # A second order system with qddot = leakyReLU(u)
    # Namely v[n+1] = v[n] + leakyReLU(u[n]) * dt
    #               = leakyReLU(v[n] + M) + leakyReLU(u[n]) * dt - M
    # where M is a larger number to make v[n] + M to be positive.
    dtype = torch.float64
    x_lo = torch.tensor([-5, -5], dtype=dtype)
    x_up = torch.tensor([5, 5], dtype=dtype)
    u_lo = torch.tensor([-10], dtype=dtype)
    u_up = torch.tensor([10], dtype=dtype)
    forward_network = utils.setup_relu(
        (3, 2, 1), params=None, negative_slope=0.1, bias=True, dtype=dtype)
    dt = 0.01
    q_shift = 20  # q_shift + q_lo should be larger than 0.
    forward_network[0].weight.data = torch.tensor(
        [[0, 1, 0], [0, 0, dt]], dtype=dtype)
    forward_network[0].bias.data = torch.tensor([q_shift, 0], dtype=dtype)
    forward_network[2].weight.data = torch.tensor([[1, 1]], dtype=dtype)
    forward_network[2].bias.data = torch.tensor([-q_shift], dtype=dtype)
    q_equilibrium = torch.tensor([0.1], dtype=dtype)
    u_equilibrium = torch.tensor([0], dtype=dtype)
    forward_system = relu_system.ReLUSecondOrderSystemGivenEquilibrium(
        dtype, x_lo, x_up, u_lo, u_up, forward_network, q_equilibrium,
        u_equilibrium, dt)
    # A linear controller PD u = K(x - x_equilibrium)
    controller_network = torch.nn.Linear(2, 1, bias=True)
    K = [-1., -10.]
    controller_network.weight.data = torch.tensor(K, dtype=dtype).reshape(
        (1, -1))
    controller_network.bias.data = -K[0] * q_equilibrium
    lyapunov_relu = utils.setup_relu(
        (2, 4, 1), params=None, negative_slope=0.01, bias=True,
        dtype=torch.float64)

    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_network, forward_system.x_equilibrium,
        u_equilibrium, u_lo.detach().numpy(), u_up.detach().numpy())

    lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(
        closed_loop_system, lyapunov_relu)

    V_lambda = 1.
    dut = train_lyapunov.TrainLyapunovReLU(
        lyap, V_lambda, forward_system.x_equilibrium)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 5E-5
    dut.max_iterations = 2000
    dut.lyapunov_positivity_epsilon = 0.5
    dut.lyapunov_derivative_epsilon = 1E-4
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.Asymp
    dut.output_flag = True
    state_samples_all = utils.get_meshgrid_samples(x_lo, x_up, (51, 51), dtype)
    dut.train_lyapunov_on_samples(state_samples_all, 100, 50)

    dut.train(torch.empty((0, 2), dtype=dtype))
    pass
