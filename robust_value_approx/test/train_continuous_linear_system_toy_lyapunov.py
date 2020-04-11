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
import time

import matplotlib.pyplot as plt
from matplotlib import cm # noqa
from mpl_toolkits import mplot3d # noqa


def setup_relu(
        relu_layer_width, params=None, negative_gradient=0.1, bias=True,
        symmetric_x=False):
    """
    @param symmetric_x If true, then we want the network satisfies
    network(x) = network(-x). This requires that bias=False, and the negative
    gradient of the first ReLU unit to be -1.
    """
    assert(isinstance(relu_layer_width, tuple))
    assert(relu_layer_width[0] == 2)
    if params is not None:
        assert(isinstance(params, torch.Tensor))
    dtype = torch.float64
    if symmetric_x:
        assert(not bias)

    def set_param(linear, param_count):
        linear.weight.data = params[
            param_count: param_count +
            linear.in_features * linear.out_features].clone().reshape((
                linear.out_features, linear.in_features))
        param_count += linear.in_features * linear.out_features
        if bias:
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
            relu_layer_width[i], next_layer_width, bias=bias).type(dtype)
        if params is None:
            pass
        else:
            param_count = set_param(linear_layers[i], param_count)
    layers = [None] * (len(relu_layer_width) * 2 - 1)
    for i in range(len(relu_layer_width) - 1):
        layers[2 * i] = linear_layers[i]
        if symmetric_x and i == 0:
            layers[2 * i + 1] = nn.LeakyReLU(-1.)
        else:
            layers[2 * i + 1] = nn.LeakyReLU(negative_gradient)
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
        help="optimizer can be either Adam, SGD or GD.")
    parser.add_argument(
        "--summary_writer_folder", type=str, default=None,
        help="folder for the tensorboard summary")
    parser.add_argument(
        "--train_on_samples_iterations", type=int, default=200,
        help="max number of iterations to pretrain on sampled states.")
    parser.add_argument(
        "--project_gradient_method", type=str, default="NONE",
        help="accept NONE, SUM, EMPHASIZE_POSITIVITY or ALTERNATE")
    parser.add_argument(
        "--momentum", type=float, default=0.,
        help="momentum in SGD and GD")
    parser.add_argument(
        "--lyapunov_positivity_sample_cost_weight", type=float, default=0.)
    parser.add_argument(
        "--lyapunov_derivative_sample_cost_weight", type=float, default=0.)
    parser.add_argument(
        "--lyapunov_derivative_mip_cost_weight", type=float, default=1.)
    parser.add_argument(
        "--lyapunov_positivity_mip_cost_weight", type=float, default=1.)
    parser.add_argument(
        "--add_adversarial_state_to_training", action="store_true")
    parser.add_argument(
        "--loss_minimal_decrement", type=float, default=None,
        help="check line_search_gd.")
    parser.add_argument(
        "--min_improvement", type=float, default=-0.1,
        help="minimal improvement in line search.")
    args = parser.parse_args()

    bias = False
    keep_symmetric_half = True
    if args.system == 1:
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system1()
        system_simulate = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system1(2.)
        x_equilibrium = torch.tensor([0., 0.], dtype=system.dtype)
        relu = setup_relu((2, 4, 2), bias=bias)
    elif args.system == 2:
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system2(
                keep_symmetric_half=keep_symmetric_half)
        system_simulate = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system2(
                10, keep_symmetric_half=False)
        x_equilibrium = torch.tensor([0., 0.], dtype=system.dtype)
        relu = setup_relu(
            (2, 4, 4, 4, 4, 4), negative_gradient=0.1, bias=bias,
            symmetric_x=True)
    elif args.system == 3:
        x_equilibrium = torch.tensor([0., 0], dtype=torch.float64)
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system3(x_equilibrium)
        system_simulate = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system3(x_equilibrium, 10)
        relu = setup_relu((2, 16, 2), bias=bias)
    elif args.system == 4:
        x_equilibrium = torch.tensor([0., 0], dtype=torch.float64)
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system4(keep_positive_x=True)
        system_simulate = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system4(10)
        relu = setup_relu(
            (2, 4, 4, 4), negative_gradient=0.1, bias=False, symmetric_x=True)
    elif args.system == 5:
        x_equilibrium = torch.tensor([0., 0], dtype=torch.float64)
        system = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system5(keep_positive_x=True)
        system_simulate = test_hybrid_linear_system.\
            setup_johansson_continuous_time_system5(10)
        relu = setup_relu(
            (2, 8, 4), negative_gradient=-1., bias=False, symmetric_x=True)

    lyapunov_hybrid_system = lyapunov.LyapunovContinuousTimeHybridSystem(
        system)

    V_rho = 0.

    if args.system in {1, 2}:
        x_lower = torch.tensor([-1, -1], dtype=system.dtype)
        x_upper = torch.tensor([1, 1], dtype=system.dtype)
    elif args.system == 3:
        x_lower = torch.tensor([-2, -1], dtype=system.dtype)
        x_upper = torch.tensor([2, 1], dtype=system.dtype)
    elif args.system in {4, 5}:
        x_lower = torch.tensor([0, -1], dtype=system.dtype)
        x_upper = torch.tensor([1, 1], dtype=system.dtype)
    if bias:
        state_samples_all = train_2d_lyapunov_utils.setup_state_samples_all(
            x_equilibrium, x_lower, x_upper, (51, 51), 0.)
    else:
        state_samples_all = train_2d_lyapunov_utils.\
            setup_state_samples_on_boundary(
                x_equilibrium, x_lower, x_upper, (51, 51), 0.)
    keep_symmetric_half = True
    # Only keep the state samples above x+y=0 line for system 2.
    if keep_symmetric_half and args.system == 2:
        state_samples_all = torch.stack(
            [state_samples_all[i, :] for i in range(state_samples_all.shape[0])
             if state_samples_all[i, 0] + state_samples_all[i, 1] >= 0])

    dut = train_lyapunov.TrainLyapunovReLU(
        lyapunov_hybrid_system, V_rho, x_equilibrium)
    dut.output_flag = True
    dut.max_iterations = args.max_iterations
    dut.learning_rate = args.learning_rate
    dut.lyapunov_positivity_mip_cost_weight = \
        args.lyapunov_positivity_mip_cost_weight
    dut.lyapunov_derivative_mip_cost_weight = \
        args.lyapunov_derivative_mip_cost_weight
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_positivity_epsilon = 0.01
    dut.lyapunov_derivative_epsilon = 0.01
    dut.lyapunov_derivative_sample_margin = 0.
    dut.lyapunov_positivity_convergence_tol = 1e-4
    dut.lyapunov_derivative_convergence_tol = 1e-4
    dut.summary_writer_folder = args.summary_writer_folder
    dut.momentum = args.momentum
    dut.loss_minimal_decrement = args.loss_minimal_decrement
    dut.min_improvement = args.min_improvement
    dut.add_adversarial_state_to_training =\
        args.add_adversarial_state_to_training
    if args.project_gradient_method == "NONE":
        dut.project_gradient_method = train_lyapunov.ProjectGradientMethod.NONE
    elif args.project_gradient_method == "SUM":
        dut.project_gradient_method = train_lyapunov.ProjectGradientMethod.SUM
    elif args.project_gradient_method == "ALTERNATE":
        dut.project_gradient_method = train_lyapunov.ProjectGradientMethod.\
            ALTERNATE
    elif args.project_gradient_method == "EMPHASIZE_POSITIVITY":
        dut.project_gradient_method = train_lyapunov.ProjectGradientMethod.\
            EMPHASIZE_POSITIVITY
    else:
        raise Exception("Unknown project gradient method.")

    if args.load_relu is None:
        if args.train_on_sample:
            # Train the Lyapunov loss on many sampled states
            relu = dut.train_lyapunov_on_samples(
                relu, state_samples_all, args.train_on_samples_iterations, 10)
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
    dut.lyapunov_positivity_sample_cost_weight = \
        args.lyapunov_positivity_sample_cost_weight
    dut.lyapunov_derivative_sample_cost_weight = \
        args.lyapunov_derivative_sample_cost_weight
    dut.lyapunov_derivative_sample_margin = 0.01
    dut.optimizer = args.optimizer

    if bias:
        state_samples = train_2d_lyapunov_utils.setup_state_samples_all(
            x_equilibrium, x_lower, x_upper, (15, 15), 0.)
    else:
        state_samples = train_2d_lyapunov_utils.\
            setup_state_samples_on_boundary(
                x_equilibrium, x_lower, x_upper, (15, 15), 0.)
    if keep_symmetric_half and args.system == 2:
        state_samples = torch.stack(
            [state_samples[i, :] for i in range(state_samples.shape[0])
             if state_samples[i, 0] + state_samples[i, 1] >= 0])
    if dut.optimizer == "GD" or dut.optimizer == "LineSearchAdam":
        result = dut.train_with_line_search(relu, state_samples)
    else:
        start_time = time.time()
        result = dut.train(relu, state_samples)
        print(f"training time: {time.time()-start_time}s")
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
