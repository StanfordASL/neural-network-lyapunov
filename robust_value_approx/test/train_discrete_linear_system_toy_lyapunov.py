# On a toy piecewise linear system, train a Lyapunov function to prove
# exponential convergence to the equilibrium

import robust_value_approx.train_lyapunov as train_lyapunov
import robust_value_approx.lyapunov as lyapunov
import robust_value_approx.test.test_hybrid_linear_system as\
    test_hybrid_linear_system
import robust_value_approx.test.train_2d_lyapunov_utils as \
    train_2d_lyapunov_utils

import torch
import torch.nn as nn

import argparse


def setup_relu(
        relu_layer_width, params=None, negative_gradient=0.1, bias=True):
    assert(isinstance(relu_layer_width, tuple))
    assert(relu_layer_width[0] == 2)
    if params is not None:
        assert(isinstance(params, torch.Tensor))
    dtype = torch.float64

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
        layers[2 * i + 1] = nn.LeakyReLU(negative_gradient)
    layers[-1] = linear_layers[-1]
    relu = nn.Sequential(*layers)
    return relu


def setup_relu1():
    # Construct a simple ReLU model with 2 hidden layers
    dtype = torch.float64
    relu_param = torch.tensor(
        [-1, 0.5, -0.3, 0.74, -2, 1.5, -0.5, 0.2, -1,
         -0.5, 1.5, 1.2, 2, -1.5, 2.6, 0.3, -2, -0.3, -.4, -0.1, 0.2, -0.5,
         1.2, 1.3, -.4, .5, -.6, 0.3], dtype=dtype)
    return setup_relu((2, 4, 4), relu_param, negative_gradient=0.1, bias=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='learning lyapunov for Trecate system parameters')
    parser.add_argument(
        '--system', type=int, default=1,
        help="1 for Trecate system, 2 for Xu system.")
    parser.add_argument(
        '--theta', type=float, default=0.,
        help='rotation angle of the Trecate system transformation')
    parser.add_argument(
        '--equilibrium_state_x', type=float, default=0.,
        help='x location of the equilibrium state')
    parser.add_argument(
        '--equilibrium_state_y', type=float, default=0.,
        help='y location of the equilibrium state')
    parser.add_argument(
        '--learning_rate', type=float, default=1e-3,
        help='learning rate for the Lyapunov function.')
    parser.add_argument(
        '--max_iterations', type=int, default=2000,
        help='max iteration for learning Lyapunov function.')
    parser.add_argument(
        '--approximator_iterations', type=int, default=1000,
        help="number of iterations to train the value function approximator")
    parser.add_argument(
        "--load_cost_to_go_data", type=str, default=None,
        help="saved pickle file on the cost-to-go samples.")
    parser.add_argument(
        "--load_relu", type=str, default=None,
        help="saved pickle file on the relu model.")
    parser.add_argument(
        '--save_model', type=str, default=None,
        help='save the Lyapunov function to a pickle file.')
    parser.add_argument(
        '--visualize', action='store_true', help='visualization flag')
    parser.add_argument(
        '--summary_writer_folder', type=str, default=None,
        help="folder for the tensorboard summary")
    args = parser.parse_args()

    theta = args.theta
    x_equilibrium = torch.tensor([
        args.equilibrium_state_x, args.equilibrium_state_y],
        dtype=torch.float64)

    if args.system == 1:
        system = test_hybrid_linear_system.setup_transformed_trecate_system(
            theta, x_equilibrium)
        system_simulate = system
        relu = setup_relu1()
    elif args.system == 2:
        system = test_hybrid_linear_system.setup_xu_system(1.)
        system_simulate = test_hybrid_linear_system.setup_xu_system(5.)
        relu = setup_relu((2, 8, 4), bias=False)

    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(system)

    V_rho = 0.1

    x_lower = torch.tensor([-1 + 1e-6, -1 + 1e-6], dtype=torch.float64)
    x_upper = torch.tensor([1 - 1e-6, 1 - 1e-6], dtype=torch.float64)

    state_samples_all1 = train_2d_lyapunov_utils.setup_state_samples_all(
        x_equilibrium, x_lower, x_upper, (51, 51), theta)
    # First train a ReLU to approximate the value function.
    approximator = train_lyapunov.TrainValueApproximator()
    approximator.max_epochs = args.approximator_iterations
    approximator.convergence_tolerance = 0.003
    if args.system == 1:
        result1 = approximator.train(
            system_simulate, relu, V_rho, x_equilibrium,
            lambda x: torch.norm(x - x_equilibrium, p=1),
            state_samples_all1, 100, True)
        print(f"value function approximation error {result1[1]}")
    elif args.system == 2:
        if args.load_relu is None:
            if args.load_cost_to_go_data is None:
                state_samples_cost_to_go = train_2d_lyapunov_utils.\
                    setup_state_samples_all(
                        x_equilibrium, x_lower, x_upper, (21, 21), theta)
                result1 = approximator.train(
                    system_simulate, relu, V_rho, x_equilibrium,
                    lambda x: 0.02 * torch.norm(x - x_equilibrium, p=1),
                    state_samples_all1, 1000, True, x_equilibrium,
                    lambda x: torch.norm(x - x_equilibrium, 1) < 0.01 and
                    torch.any(x - x_equilibrium <= x_lower) and
                    torch.any(x - x_equilibrium >= x_upper))
            else:
                x0_value_samples = torch.load(args.load_cost_to_go_data)
                result1 = approximator.train_with_cost_to_go(
                    relu, x0_value_samples, V_rho, x_equilibrium)

            print(f"value function approximation error {result1[1]}")
        else:
            relu = torch.load(args.load_relu)

    state_samples_all = state_samples_all1
    dut = train_lyapunov.TrainLyapunovReLU(
        lyapunov_hybrid_system, V_rho, x_equilibrium)
    dut.output_flag = True
    dut.max_iterations = args.max_iterations
    dut.learning_rate = args.learning_rate
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_positivity_sample_cost_weight = 0.
    dut.lyapunov_derivative_sample_cost_weight = 0.
    if args.system == 1:
        dut.lyapunov_positivity_mip_cost_weight = 0.
    elif args.system == 2:
        dut.lyapunov_positivity_mip_cost_weight = 1.
    dut.lyapunov_positivity_convergence_tol = 1e-5
    if args.system == 1:
        dut.lyapunov_derivative_convergence_tol = 4e-5
    elif args.system == 2:
        dut.lyapunov_derivative_convergence_tol = 8e-5
    dut.summary_writer_folder = args.summary_writer_folder
    if (args.visualize):
        train_2d_lyapunov_utils.plot_relu(
            relu, system, V_rho, dut.lyapunov_positivity_epsilon,
            dut.lyapunov_derivative_epsilon, x_equilibrium, x_lower, x_upper,
            (51, 51), theta, discrete_time=True)
    result = dut.train(relu, state_samples_all)

    if args.save_model is not None:
        lyapunov = {
            "relu": relu,
            "V_rho": V_rho,
            "x_equilibrium": x_equilibrium,
            "theta": theta,
            "lyapunov_positivity_epsilon": dut.lyapunov_positivity_epsilon,
            "lyapunov_derivative_epsilon": dut.lyapunov_derivative_epsilon}
        torch.save(lyapunov, args.save_model)
    if (args.visualize):
        train_2d_lyapunov_utils.plot_relu(
            relu, system, V_rho, dut.lyapunov_positivity_epsilon,
            dut.lyapunov_derivative_epsilon, x_equilibrium, x_lower, x_upper,
            (51, 51), theta, discrete_time=True)
