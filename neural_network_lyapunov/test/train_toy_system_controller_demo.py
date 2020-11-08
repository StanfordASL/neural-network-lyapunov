import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.relu_system as relu_system
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="parse train controller on toy problems")
    parser.add_argument(
        "--dimension", type=int, default=2, help="either 1 or 2")
    args = parser.parse_args()

    if args.dimension == 1:
        linear_layer_width = (2, 4, 1)
        params = torch.tensor(
            [1, -1, -1, 2, 0.5, 0.5, -2, 1, 0.5, -0.5, -1, 1, 1, -1, 2, -2,
             0.5], dtype=torch.float64)
        x_equilibrium = torch.tensor([1], dtype=torch.float64)
        u_equilibrium = torch.tensor([0], dtype=torch.float64)
        controller_linear_layer_width = (1, 4, 4, 1)
        lyapunov_linear_layer_width = (1, 5, 5, 1)
        x_lo = torch.tensor([0.5], dtype=torch.float64)
        x_up = torch.tensor([1.5], dtype=torch.float64)
        u_lo = torch.tensor([-10], dtype=torch.float64)
        u_up = torch.tensor([10], dtype=torch.float64)
        state_samples_all = torch.tensor([[0.5], [1.5]], dtype=torch.float64)
    elif args.dimension == 2:
        linear_layer_width = (4, 4, 2)
        params = torch.tensor([
            1, -1, 0, 1, -1, 2, 1, -1, 0.5, 0.5, 0, 0.5, -2, 1, -0.5, -0.5,
            0.5, -0.5, -1, 1, 1, -1, 2, -2, -0.5, 2, 1, -1, 0.5, -1],
            dtype=torch.float64)
        x_equilibrium = torch.tensor([1, 0], dtype=torch.float64)
        u_equilibrium = torch.tensor([0, 0], dtype=torch.float64)
        controller_linear_layer_width = (2, 6, 4, 2)
        lyapunov_linear_layer_width = (2, 5, 5, 3, 1)
        x_lo = torch.tensor([0.5, -2], dtype=torch.float64)
        x_up = torch.tensor([1.5, 2], dtype=torch.float64)
        u_lo = torch.tensor([-10, -10], dtype=torch.float64)
        u_up = torch.tensor([10, 10], dtype=torch.float64)
        state_samples_all = torch.tensor(
            [[0.5, 1], [1.5, 1], [0.5, 0], [1.2, 0]], dtype=torch.float64)

    forward_relu = utils.setup_relu(
        linear_layer_width, params=params, negative_slope=0.01, bias=True,
        dtype=torch.float64)

    V_lambda = 0.1
    controller_relu = utils.setup_relu(
        controller_linear_layer_width, params=None, negative_slope=0.01,
        bias=True, dtype=torch.float64)
    lyapunov_relu = utils.setup_relu(
        lyapunov_linear_layer_width, params=None, negative_slope=0.01,
        bias=True, dtype=torch.float64)

    forward_system = relu_system.ReLUSystemGivenEquilibrium(
        torch.float64, x_lo, x_up, u_lo, u_up, forward_relu, x_equilibrium,
        u_equilibrium)

    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, x_equilibrium, u_equilibrium,
        u_lo.detach().numpy(), u_up.detach().numpy())

    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
        closed_loop_system, lyapunov_relu)

    R = None
    dut = train_lyapunov.TrainLyapunovReLU(
        lyapunov_hybrid_system, V_lambda, x_equilibrium, R)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 5E-5
    dut.max_iterations = 3000
    dut.lyapunov_positivity_epsilon = 0.05
    dut.lyapunov_derivative_epsilon = 0.01
    dut.output_flag = True
    dut.train(state_samples_all)
    pass
