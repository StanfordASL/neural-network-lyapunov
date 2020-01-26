import robust_value_approx.train_lyapunov as train_lyapunov
import robust_value_approx.lyapunov as lyapunov
import robust_value_approx.test.test_hybrid_linear_system as\
    test_hybrid_linear_system

import torch
import torch.nn as nn
import numpy as np

import argparse

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d # noqa


def setup_relu():
    # Construct a simple ReLU model with 2 hidden layers
    dtype = torch.float64
    linear1 = nn.Linear(2, 4)
    linear1.weight.data = torch.tensor(
        [[-1, 0.5], [-0.3, 0.74], [-2, 1.5], [-0.5, 0.2]], dtype=dtype)
    linear1.bias.data = torch.tensor(
        [-0.1, 1.0, 0.5, 0.2], dtype=dtype)
    linear2 = nn.Linear(4, 4)
    linear2.weight.data = torch.tensor(
            [[-1, -0.5, 1.5, 1.2], [2, -1.5, 2.6, 0.3], [-2, -0.3, -.4, -0.1],
             [0.2, -0.5, 1.2, 1.3]],
            dtype=dtype)
    linear2.bias.data = torch.tensor([-0.3, 0.2, 0.7, 0.4], dtype=dtype)
    linear3 = nn.Linear(4, 1)
    linear3.weight.data = torch.tensor(
        [[-.4, .5, -.6, 0.3]], dtype=dtype)
    linear3.bias.data = torch.tensor([-0.9], dtype=dtype)
    relu1 = nn.Sequential(
        linear1, nn.LeakyReLU(0.1), linear2, nn.LeakyReLU(0.1), linear3,
        nn.LeakyReLU(0.1))
    return relu1

def setup_state_samples_all(lower, upper, mesh_size):
    assert(isinstance(lower, torch.Tensor))
    assert(isinstance(upper, torch.Tensor))
    assert(lower.shape == (2,))
    assert(upper.shape == (2,))
    assert(len(mesh_size) == 2)
    dtype = torch.float64
    (samples_x, samples_y) = torch.meshgrid(
        torch.linspace(lower[0], upper[0], mesh_size[0], dtype=dtype),
        torch.linspace(lower[1], upper[1], mesh_size[1], dtype=dtype))
    state_samples = [None] * (mesh_size[0] * mesh_size[1])
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            state_samples[i * mesh_size[1] + j] = torch.tensor(
                [samples_x[i, j], samples_y[i, j]], dtype=dtype)
    return state_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="learning lyapunov for toy continuous time hybrid " +
        "linear system.")
    parser.add_argument(
        "--system", type=int, default=1,
        help="the system index.")
    parser.add_argument(
        '--learning_rate', type=float, default=1e-3,
        help='learning rate for the Lyapunov function.')
    parser.add_argument(
        "--max_iterations", type=int, default=2000,
        help="max iteration for learning Lyapunov function.")
    args = parser.parse_args()

    if args.system == 1:
        system = test_hybrid_linear_system.setup_johansson_continuous_time_system1()
    elif args.system == 2:
        system = test_hybrid_linear_system.setup_johansson_continuous_time_system2()
    elif args.system == 3:
        system = test_hybrid_linear_system.setup_johansson_continuous_time_system3()

    lyapunov_hybrid_system = lyapunov.LyapunovContinuousTimeHybridSystem(
        system)

    relu = setup_relu()
    V_rho = 0.1

    if args.system == 1 or argas.system == 2:
        state_samples_all = setup_state_samples_all(
            torch.tensor([-1., -1.], dtype=system.dtype),
            torch.tensor([1., 1.], dtype=system.dtype), (51, 51))
    elif args.system == 3:
        state_samples_all = setup_state_samples_all(
            torch.tensor([-2., -1.], dtype=system.dtype),
            torch.tensor([2., 1.], dtype=system.dtype), (51, 51))

    x_equilibrium = torch.tensor([0., 0.], dtype=system.dtype)

    # First train a ReLU to approximate the value function.
    approximator = train_lyapunov.TrainValueApproximator()
    approximator.max_epochs = 10
    approximator.convergence_tolerance = 0.01
    result1 = approximator.train(
        system, relu,V_rho, x_equilibrium,
        lambda x: torch.norm(x - x_equilibrium, p=1), state_samples_all, 5.,
        False)

    dut = train_lyapunov.TrainLyapunovReLU(
        lyapunov_hybrid_system, V_rho, x_equilibrium)
    dut.output_flag = True
    dut.max_iterations = args.max_iterations
    dut.learning_rate = args.learning_rate
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_positivity_sample_cost_weight = 0.
    dut.lyapunov_derivative_sample_cost_weight = 0.
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_positivity_mip_cost_weight = 1.
    result = dut.train(relu, state_samples_all)
