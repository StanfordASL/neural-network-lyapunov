# On a toy piecewise linear system, train a Lyapunov function to prove
# exponential convergence to the equilibrium

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


def setup_state_samples_all(mesh_size, x_equilibrium, theta):
    assert(isinstance(mesh_size, tuple))
    assert(len(mesh_size) == 2)
    dtype = torch.float64
    (samples_x, samples_y) = torch.meshgrid(
        torch.linspace(-1.+1e-6, 1.-1e-6, mesh_size[0], dtype=dtype),
        torch.linspace(-1.+1e-6, 1.-1e-6, mesh_size[1], dtype=dtype))
    state_samples = [None] * (mesh_size[0] * mesh_size[1])
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = torch.tensor([[
        cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=dtype)
    for i in range(samples_x.shape[0]):
        for j in range(samples_x.shape[1]):
            state_samples[i * samples_x.shape[1] + j] = R @ torch.tensor(
                [samples_x[i, j], samples_y[i, j]], dtype=dtype) +\
                    x_equilibrium
    return torch.stack(state_samples, dim=0)


def plot_relu(relu, system, V_rho, x_equilibrium, mesh_size, theta):
    assert(isinstance(mesh_size, tuple))
    assert(len(mesh_size) == 2)
    dtype = torch.float64
    assert(isinstance(theta, float))
    with torch.no_grad():
        state_samples_all = setup_state_samples_all(
            mesh_size, x_equilibrium, theta)
        samples_x = torch.empty(mesh_size)
        samples_y = torch.empty(mesh_size)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                samples_x[i, j] = state_samples_all[i * mesh_size[1] + j][0]
                samples_y[i, j] = state_samples_all[i * mesh_size[1] + j][1]
        V = torch.zeros(mesh_size)
        dV = torch.zeros(mesh_size)
        relu_at_equilibrium = relu.forward(x_equilibrium)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                state_sample = torch.tensor(
                    [samples_x[i, j], samples_y[i, j]], dtype=dtype)
                V[i, j] = relu.forward(state_sample) - relu_at_equilibrium +\
                    V_rho * torch.norm(state_sample - x_equilibrium, p=1)
                state_next = None
                mode = system.mode(state_sample)
                state_next = system.step_forward(state_sample, mode)
                V_next = relu.forward(state_next) - relu_at_equilibrium +\
                    V_rho * torch.norm(state_next - x_equilibrium, p=1)
                dV[i, j] = V_next - V[i, j]

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1, projection='3d')
    ax1.plot_surface(samples_x.detach().numpy(),
                     samples_y.detach().numpy(), V.detach().numpy(),
                     cmap=cm.coolwarm)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("V")
    ax1.set_title("Lyapunov function")

    ax2 = fig.add_subplot(3, 1, 2)
    plot2 = ax2.pcolor(samples_x.detach().numpy(), samples_y.detach().numpy(),
                       V.detach().numpy())
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("V(x[n])")
    fig.colorbar(plot2, ax=ax2)

    ax3 = fig.add_subplot(3, 1, 3)
    plot3 = ax3.pcolor(samples_x.detach().numpy(), samples_y.detach().numpy(),
                       dV.detach().numpy())
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("V(x[n+1])-V(x[n])")
    fig.colorbar(plot3, ax=ax3)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='learning lyapunov for Trecate system parameters')
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

    system = test_hybrid_linear_system.setup_transformed_trecate_system(
        theta, x_equilibrium)
    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(system)

    relu = setup_relu()
    V_rho = 0.1

    state_samples_all1 = setup_state_samples_all(
        (51, 51), x_equilibrium, theta)
    # First train a ReLU to approximate the value function.
    approximator = train_lyapunov.TrainValueApproximator()
    approximator.max_epochs = args.approximator_iterations
    approximator.convergence_tolerance = 0.003
    result1 = approximator.train(
        system, relu, V_rho, x_equilibrium,
        lambda x: torch.norm(x - x_equilibrium, p=1),
        state_samples_all1, 100, True)
    print(f"value function approximation error {result1[1]}")
    if (args.visualize):
        plot_relu(relu, system, V_rho, x_equilibrium, (51, 51), theta)

    state_samples_all = state_samples_all1
    dut = train_lyapunov.TrainLyapunovReLU(
        lyapunov_hybrid_system, V_rho, x_equilibrium)
    dut.output_flag = True
    dut.max_iterations = args.max_iterations
    dut.learning_rate = args.learning_rate
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_positivity_sample_cost_weight = 0.
    dut.lyapunov_derivative_sample_cost_weight = 0.
    dut.lyapunov_positivity_mip_cost_weight = 0.
    dut.lyapunov_positivity_convergence_tol = 1e-5
    dut.lyapunov_derivative_convergence_tol = 4e-5
    dut.summary_writer_folder = args.summary_writer_folder
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
        plot_relu(relu, system, V_rho, x_equilibrium, (51, 51), theta)
