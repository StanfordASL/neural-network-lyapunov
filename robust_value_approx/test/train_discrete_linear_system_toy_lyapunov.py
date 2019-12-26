# On a toy piecewise linear system, train a Lyapunov function to prove
# exponential convergence to the equilibrium

import robust_value_approx.train_lyapunov as train_lyapunov
import robust_value_approx.lyapunov as lyapunov
import torch
import torch.nn as nn
import robust_value_approx.test.test_hybrid_linear_system as\
    test_hybrid_linear_system
import robust_value_approx.test.test_train_lyapunov as test_train_lyapunov

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d


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
    linear3 = nn.Linear(4, 4)
    linear3.weight.data = torch.tensor(
        [[-0.5, 0.2, 0.3, 0.1], [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, -0.1, -0.2],
         [0.3, -1.1, 0.2, 0.4]], dtype=dtype)
    linear3.bias.data = torch.tensor([-0.1, 0.1, 0.2, 0.3], dtype=dtype)
    linear4 = nn.Linear(4, 1)
    linear4.weight.data = torch.tensor(
        [[-.4, .5, -.6, 0.3]], dtype=dtype)
    linear4.bias.data = torch.tensor([-0.9], dtype=dtype)
    relu1 = nn.Sequential(
        linear1, nn.LeakyReLU(0.1), linear4, nn.LeakyReLU(0.1))#, linear2, nn.LeakyReLU(0.1), linear3,
        #nn.LeakyReLU(0.1), linear4, nn.LeakyReLU(0.1))
    return relu1


def plot_relu(relu, system, V_rho, x_equilibrium, mesh_size):
    assert(isinstance(mesh_size, tuple))
    assert(len(mesh_size) == 2)
    dtype = torch.float64
    with torch.no_grad():
        (samples_x, samples_y) = torch.meshgrid(
            torch.linspace(-1., 1., mesh_size[0], dtype=dtype),
            torch.linspace(-1., 1., mesh_size[1], dtype=dtype))
        V = torch.zeros(mesh_size)
        dV = torch.zeros(mesh_size)
        relu_at_equilibrium = relu.forward(x_equilibrium)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                state_sample = torch.tensor(
                    [samples_x[i, j], samples_y[i, j]], dtype=dtype)
                V[i, j] = relu.forward(state_sample) - relu_at_equilibrium +\
                    V_rho * torch.norm(state_sample - x_equilibrium, p=1)
                for mode in range(system.num_modes):
                    if (torch.all(system.P[mode] @ state_sample <=
                                  system.q[mode])):
                        state_next = system.A[mode] @ state_sample +\
                            system.g[mode]
                        break
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
    system = test_hybrid_linear_system.setup_trecate_discrete_time_system()
    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(system)

    relu = setup_relu()
    V_rho = 0.
    x_equilibrium = torch.tensor([0, 0], dtype=torch.float64)

    state_samples_all1 = test_train_lyapunov.setup_state_samples_all((51, 51))
    # First train a ReLU to approximate the value function.
    options1 = train_lyapunov.TrainValueApproximatorOptions()
    options1.max_epochs = 200
    options1.convergence_tolerance = 0.01
    result1 = train_lyapunov.train_value_approximator(
        system, relu, V_rho, x_equilibrium, lambda x: torch.norm(x, p=1),
        state_samples_all1, options1)
    plot_relu(relu, system, V_rho, x_equilibrium, (51, 51))

    state_samples_all = state_samples_all1
    dut = train_lyapunov.TrainLyapunovReLU(lyapunov_hybrid_system, V_rho, x_equilibrium)
    dut.output_flag = True
    dut.max_iterations = 1000
    dut.lyapunov_derivative_mip_pool_solutions = 20
    result = dut.train(relu, state_samples_all)

#    result = train_lyapunov.train_lyapunov_relu(
#        lyapunov_hybrid_system, relu, V_rho, x_equilibrium, state_samples_all,
#        options)
    plot_relu(relu, system, V_rho, x_equilibrium, (51, 51))
