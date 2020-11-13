import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.train_lyapunov as train_lyapunov

import torch
import numpy as np


def create_system(dtype):
    # Create a PWL hybrid system.
    system = hybrid_linear_system.HybridLinearSystem(2, 2, dtype)
    # The system has two modes
    # x[n+1] = x[n] + u[n] if [0, -10, -10, -10] <= [x[n], u[n]] <=
    # [10, 10, 10, 10]
    # x[n+1] = 0.5x[n] + 1.5u[n] if [-10, -10, -10, -10] <= [x[n], u[n]] <=
    # [0, 10, 10, 10]
    P = torch.zeros(8, 4, dtype=dtype)
    P[:4, :] = torch.eye(4, dtype=dtype)
    P[4:, :] = -torch.eye(4, dtype=dtype)
    system.add_mode(
        torch.eye(2, dtype=dtype), torch.eye(2, dtype=dtype),
        torch.tensor([0, 0], dtype=dtype), P,
        torch.tensor([10, 10, 10, 10, 0, 10, 10, 10], dtype=dtype))
    system.add_mode(
        0.5*torch.eye(2, dtype=dtype), 1.5*torch.eye(2, dtype=dtype),
        torch.tensor([0, 0], dtype=dtype), P,
        torch.tensor([0, 10, 10, 10, 10, 10, 10, 10], dtype=dtype))
    return system


if __name__ == "__main__":
    dtype = torch.float64
    forward_system = create_system(dtype)

    controller_network = utils.setup_relu(
        (2, 4, 2), None, negative_slope=0.01, bias=False, dtype=dtype)

    x_equilibrium = torch.tensor([0, 0], dtype=dtype)
    u_equilibrium = torch.tensor([0, 0], dtype=dtype)
    u_lower_limit = np.array([-20., -20.])
    u_upper_limit = np.array([20., 20.])

    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_network, x_equilibrium, u_equilibrium,
        u_lower_limit, u_upper_limit)

    lyapunov_relu = utils.setup_relu(
        (2, 4, 4, 1), None, negative_slope=0.01, bias=False, dtype=dtype)
    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
        closed_loop_system, lyapunov_relu)

    V_lambda = 0.0
    R = None
    dut = train_lyapunov.TrainLyapunovReLU(
        lyapunov_hybrid_system, V_lambda, x_equilibrium, R)
    dut.output_flag = True
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.max_iterations = 2000

    state_samples_all = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=dtype)
    result = dut.train(state_samples_all)

    pass
