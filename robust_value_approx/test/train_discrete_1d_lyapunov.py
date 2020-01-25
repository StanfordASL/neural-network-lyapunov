import robust_value_approx.train_lyapunov as train_lyapunov
import robust_value_approx.lyapunov as lyapunov
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


def setup_system():
    """
    Test a simple linear hybrid system
    x[n+1] = -0.9 * x[n] if 0 <= x[n] <= 1
    x[n+1] = -0.5 * x[n] if -1 <= x[n] <= 0
    """
    dtype = torch.float64
    sys = hybrid_linear_system.AutonomousHybridLinearSystem(1, dtype)
    sys.add_mode(
        torch.tensor([[-0.9]], dtype=dtype), torch.tensor([0], dtype=dtype),
        torch.tensor([[1], [-1]], dtype=dtype),
        torch.tensor([1, 0], dtype=dtype))
    sys.add_mode(
        torch.tensor([[-0.5]], dtype=dtype), torch.tensor([0], dtype=dtype),
        torch.tensor([[-1], [1]], dtype=dtype),
        torch.tensor([1, 0], dtype=dtype))
    return sys


def setup_relu():
    dtype = torch.float64
    linear1 = nn.Linear(1, 4)
    linear1.weight.data = torch.tensor(
        [[0], [0.1], [-0.1], [0.2]], dtype=dtype)
    linear1.bias.data = torch.tensor(
        [0.1, 0.01, -0.1, -0.01], dtype=dtype)
    linear2 = nn.Linear(4, 1)
    linear2.weight.data = torch.tensor([[0.1, 0.2, -0.1, -0.2]], dtype=dtype)
    linear2.bias.data = torch.tensor([0.1], dtype=dtype)
    relu = nn.Sequential(linear1, nn.LeakyReLU(0.1), linear2,
                         nn.LeakyReLU(0.1))
    return relu


def plot_relu(relu, system, V_rho, x_equilibrium):
    num_samples = 1001
    x_samples = list(torch.linspace(-1, 1, num_samples).type(torch.float64).
                     reshape((-1, 1)))
    x_next = [system.step_forward(x) for x in x_samples]
    with torch.no_grad():
        V = torch.zeros(num_samples)
        V_next = torch.zeros(num_samples)
        dV = torch.zeros(num_samples)
        relu_at_equilibrium = relu.forward(x_equilibrium)
        for i in range(num_samples):
            V[i] = relu.forward(x_samples[i]) - relu_at_equilibrium +\
                V_rho * torch.norm(x_samples[i] - x_equilibrium, p=1)
            V_next[i] = relu.forward(x_next[i]) - relu_at_equilibrium +\
                V_rho * torch.norm(x_next[i] - x_equilibrium, p=1)
            dV[i] = V_next[i] - V[i]
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    x_samples_numpy = np.array([x.detach().numpy() for x in x_samples])
    ax1.plot(x_samples_numpy, V.detach().numpy())
    ax1.plot(np.array([-1., 1.]), np.array([0., 0.]), '--')
    ax1.set_ylabel("V")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x_samples_numpy, dV.detach().numpy())
    ax2.plot(np.array([-1., 1.]), np.array([0., 0.]), '--')
    ax2.set_ylabel("dV")
    plt.show()


if __name__ == "__main__":
    system = setup_system()
    relu = setup_relu()
    V_rho = 0.1
    x_equilibrium = torch.tensor([0], dtype=torch.float64)
    state_samples_all = list(torch.linspace(-1, 1, 20).type(torch.float64).
                             reshape((-1, 1)))
    train_value_approximator = train_lyapunov.TrainValueApproximator()
    train_value_approximator.max_epochs = 500
    train_value_approximator.convergence_tolerance = 0.001
    result1 = train_value_approximator.train(
        system, relu, V_rho, x_equilibrium, lambda x: torch.norm(x, p=1),
        state_samples_all, 100, True)
    plot_relu(relu, system, V_rho, x_equilibrium)

    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(system)
    dut = train_lyapunov.TrainLyapunovReLU(
        lyapunov_hybrid_system, V_rho, x_equilibrium)
    dut.output_flag = True
    dut.max_iterations = 3000
    dut.learning_rate = 1e-4
    result = dut.train(relu, state_samples_all)
    plot_relu(relu, system, V_rho, x_equilibrium)
