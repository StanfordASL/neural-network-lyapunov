import robust_value_approx.train_lyapunov as train_lyapunov
import robust_value_approx.lyapunov as lyapunov
import robust_value_approx.test.test_hybrid_linear_system as\
    test_hybrid_linear_system

import torch
import torch.nn as nn
import numpy as np

import argparse

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # noqa


def setup_relu(params=None):
    # Construct a simple ReLU model with 2 hidden layers
    if params is not None:
        assert(isinstance(params, torch.Tensor))
        assert(params.shape == (37,))
    dtype = torch.float64
    linear1 = nn.Linear(2, 4)
    if params is None:
        linear1.weight.data = torch.tensor(
            [[-1, 0.5], [-0.3, 0.74], [-2, 1.5], [-0.5, 0.2]], dtype=dtype)
        linear1.bias.data = torch.tensor(
            [-0.1, 1.0, 0.5, 0.2], dtype=dtype)
    else:
        linear1.weight.data = params[:8].clone().reshape((4, 2))
        linear1.bias.data = params[8:12].clone()
    linear2 = nn.Linear(4, 4)
    if params is None:
        linear2.weight.data = torch.tensor(
            [[-1, -0.5, 1.5, 1.2], [2, -1.5, 2.6, 0.3], [-2, -0.3, -.4, -0.1],
             [0.2, -0.5, 1.2, 1.3]],
            dtype=dtype)
        linear2.bias.data = torch.tensor([-0.3, 0.2, 0.7, 0.4], dtype=dtype)
    else:
        linear2.weight.data = params[12:28].clone().reshape((4, 4))
        linear2.bias.data = params[28:32].clone()
    linear3 = nn.Linear(4, 1)
    if params is None:
        linear3.weight.data = torch.tensor(
            [[-.4, .5, -.6, 0.3]], dtype=dtype)
        linear3.bias.data = torch.tensor([-0.9], dtype=dtype)
    else:
        linear3.weight.data = params[32:36].clone().reshape((1, 4))
        linear3.bias.data = params[36].clone().reshape((1))
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
    parser.add_argument(
        "--visualize", help="visualize the results", action="store_true")
    parser.add_argument(
        "--approximator_iterations", type=int, default=100,
        help="number of iterations for training a value function approximator")
    args = parser.parse_args()

    if args.system == 1:
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system1()
        x_equilibrium = torch.tensor([0., 0.], dtype=system.dtype)
    elif args.system == 2:
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system2()
        x_equilibrium = torch.tensor([0., 0.], dtype=system.dtype)
    elif args.system == 3:
        x_equilibrium = torch.tensor([0.5, 0.3], dtype=torch.float64)
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system3(x_equilibrium)

    lyapunov_hybrid_system = lyapunov.LyapunovContinuousTimeHybridSystem(
        system)

    relu = setup_relu()
    V_rho = 0.1

    if args.system == 1 or args.system == 2:
        state_samples_all = setup_state_samples_all(
            torch.tensor([-1., -1.], dtype=system.dtype),
            torch.tensor([1., 1.], dtype=system.dtype), (51, 51))
    elif args.system == 3:
        state_samples_all = setup_state_samples_all(
            torch.tensor([-2., -1.], dtype=system.dtype),
            torch.tensor([2., 1.], dtype=system.dtype), (51, 51))

    # First train a ReLU to approximate the value function.
    approximator = train_lyapunov.TrainValueApproximator()
    approximator.max_epochs = args.approximator_iterations
    approximator.convergence_tolerance = 0.01
    result1 = approximator.train(
        system, relu, V_rho, x_equilibrium,
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
    if args.visualize:
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(np.log10(np.array(result[1])))
        ax1.set_title("loss")

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(np.log10(np.array(result[2])))
        ax2.set_title("min V(x) - epsilon * |x-x*|")

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(np.log10(np.array(result[3])))
        ax3.set_title("max Vdot + epsilon * V")
        ax3.set_xlabel("iteration count")

        plt.show()
