import robust_value_approx.train_lyapunov as train_lyapunov
import robust_value_approx.lyapunov as lyapunov
import robust_value_approx.test.test_hybrid_linear_system as\
    test_hybrid_linear_system
import robust_value_approx.test.train_2d_lyapunov_utils as\
    train_2d_lyapunov_utils

import torch
import torch.nn as nn
import numpy as np

import argparse

import matplotlib.pyplot as plt
from matplotlib import cm # noqa
from mpl_toolkits import mplot3d # noqa


def setup_relu(relu_layer_width, params=None):
    assert(isinstance(relu_layer_width, tuple))
    assert(relu_layer_width[0] == 2)
    # Construct a simple ReLU model with 2 hidden layers
    if params is not None:
        assert(isinstance(params, torch.Tensor))
    dtype = torch.float64

    def set_param(linear, param_count):
        linear.weight.data = params[
            param_count: param_count +
            linear.in_features * linear.out_features].clone().reshape((
                linear.out_features, linear.in_features))
        param_count += linear.in_features * linear.out_features
        linear.bias.data = params[
            param_count: param_count + linear.out_features].clone()
        param_count += linear.out_features
        return param_count

    linear_layers = [None] * len(relu_layer_width)
    param_count = 0
    for i in range(len(relu_layer_width)):
        next_layer_width = relu_layer_width[i+1] if \
            i < len(relu_layer_width)-1 else 1
        linear_layers[i] = nn.Linear(
            relu_layer_width[i], next_layer_width).type(dtype)
        if params is None:
            pass
        else:
            param_count = set_param(linear_layers[i], param_count)
    layers = [None] * (len(relu_layer_width) * 2 - 1)
    for i in range(len(relu_layer_width) - 1):
        layers[2 * i] = linear_layers[i]
        layers[2 * i + 1] = nn.LeakyReLU(0.2)
    layers[-1] = linear_layers[-1]
    relu = nn.Sequential(*layers)
    return relu


if __name__ == "__main__":
    plt.ion()
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
        "--train_on_sample", help="Train candidate Lyapunov on sampled states",
        action="store_true")
    parser.add_argument(
        "--approximator_iterations", type=int, default=100,
        help="number of iterations for training a value function approximator")
    parser.add_argument(
        "--load_cost_to_go_data", type=str, default=None,
        help="saved pickle file on the cost-to-go samples.")
    parser.add_argument(
        "--load_relu", type=str, default=None,
        help="saved pickle file on the relu model.")
    parser.add_argument(
        "--optimizer", type=str, default="Adam",
        help="optimizer can be either Adam or SGD.")
    args = parser.parse_args()

    if args.system == 1:
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system1()
        system_simulate = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system1(2.)
        x_equilibrium = torch.tensor([0., 0.], dtype=system.dtype)
        relu = setup_relu((2, 4, 2))
    elif args.system == 2:
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system2()
        system_simulate = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system2(10)
        x_equilibrium = torch.tensor([0., 0.], dtype=system.dtype)
        relu = setup_relu((2, 8, 4))
    elif args.system == 3:
        x_equilibrium = torch.tensor([0., 0], dtype=torch.float64)
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system3(x_equilibrium)
        system_simulate = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system3(x_equilibrium, 10)
        relu = setup_relu((2, 8, 4))

    lyapunov_hybrid_system = lyapunov.LyapunovContinuousTimeHybridSystem(
        system)

    V_rho = 0.02

    if args.system == 1 or args.system == 2:
        x_lower = torch.tensor([-1, -1], dtype=system.dtype)
        x_upper = torch.tensor([1, 1], dtype=system.dtype)
    elif args.system == 3:
        x_lower = torch.tensor([-2, -1], dtype=system.dtype)
        x_upper = torch.tensor([2, 1], dtype=system.dtype)
    state_samples_all = train_2d_lyapunov_utils.setup_state_samples_all(
        x_equilibrium, x_lower, x_upper, (51, 51), 0.)

    dut = train_lyapunov.TrainLyapunovReLU(
        lyapunov_hybrid_system, V_rho, x_equilibrium)
    dut.output_flag = True
    dut.max_iterations = args.max_iterations
    dut.learning_rate = args.learning_rate
    dut.lyapunov_positivity_mip_cost_weight = 1.
    dut.lyapunov_derivative_mip_cost_weight = 1.
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_positivity_epsilon = 0.001
    dut.lyapunov_derivative_epsilon = 0.001
    dut.lyapunov_derivative_sample_margin = 0.
    dut.lyapunov_positivity_convergence_tol = 1e-4
    dut.lyapunov_derivative_convergence_tol = 1e-4

    if args.load_relu is None:
        if args.train_on_sample:
            # Train the Lyapunov loss on many sampled states
            relu = dut.train_lyapunov_on_samples(
                relu, state_samples_all, 200, 10)
        else:
            # First train a ReLU to approximate the value function.
            approximator = train_lyapunov.TrainValueApproximator()
            approximator.max_epochs = args.approximator_iterations
            approximator.convergence_tolerance = 1e-5
            if args.load_cost_to_go_data is None:
                result1, loss1 = approximator.train(
                    system_simulate, relu, V_rho, x_equilibrium,
                    lambda x: 0.1 * torch.norm(x - x_equilibrium, p=2),
                    state_samples_all, 100., False, x_equilibrium,
                    lambda x: torch.norm(x - x_equilibrium, p=2) < 0.01 and
                    torch.any(x - x_equilibrium <= x_lower) and
                    torch.any(x - x_equilibrium >= x_upper))
            else:
                x0_value_samples = torch.load(args.load_cost_to_go_data)
                result1, loss1 = approximator.train_with_cost_to_go(
                    relu, x0_value_samples, V_rho, x_equilibrium)
                train_2d_lyapunov_utils.plot_cost_to_go_approximator(
                    relu, x0_value_samples, x_equilibrium, V_rho, x_lower,
                    x_upper, (101, 101), 0.)
            print(f"approximator loss {loss1}")
    else:
        relu = torch.load(args.load_relu)

    # No loss on sampled states. Only use MIP loss.
    dut.lyapunov_positivity_sample_cost_weight = 0.
    dut.lyapunov_derivative_sample_cost_weight = 1.
    dut.optimizer = args.optimizer

    state_samples = train_2d_lyapunov_utils.setup_state_samples_all(
        x_equilibrium, x_lower, x_upper, (15, 15), 0.)
    result = dut.train(relu, state_samples)
    if args.visualize:
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(np.log10(np.array(result[1])))
        ax1.set_title("loss")

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(np.log10(-np.array(result[2])))
        ax2.set_title("-min V(x) - epsilon * |x-x*|")

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(np.log10(np.array(result[3])))
        ax3.set_title("max Vdot + epsilon * V")
        ax3.set_xlabel("iteration count")

        plt.show()
