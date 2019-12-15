import robust_value_approx.train_lyapunov as train_lyapunov
import robust_value_approx.lyapunov as lyapunov
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import robust_value_approx.test.test_hybrid_linear_system as\
    test_hybrid_linear_system
import robust_value_approx.test.test_train_lyapunov as test_train_lyapunov


def plot_relu(relu, system, mesh_size):
    assert(isinstance(mesh_size, tuple))
    assert(len(mesh_size) == 2)
    dtype = torch.float64
    with torch.no_grad():
        (samples_x, samples_y) = torch.meshgrid(
            torch.linspace(-1., 1., mesh_size[0], dtype=dtype),
            torch.linspace(-1., 1., mesh_size[1], dtype=dtype))
        V = torch.zeros(mesh_size)
        dV = torch.zeros(mesh_size)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                state_sample = torch.tensor(
                    [samples_x[i, j], samples_y[i, j]], dtype=dtype)
                V[i, j] = relu.forward(state_sample)
                for mode in range(system.num_modes):
                    if (torch.all(system.P[mode] @ state_sample <=
                                  system.q[mode])):
                        state_next = system.A[mode] @ state_sample +\
                            system.g[mode]
                        break
                V_next = relu.forward(state_next)
                dV[i, j] = V_next - V[i, j]

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    ax1.plot_surface(samples_x.detach().numpy(),
                     samples_y.detach().numpy(), V.detach().numpy(),
                     cmap=cm.coolwarm)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("V")
    ax1.set_title("Lyapunov function")
    ax2 = fig.add_subplot(2, 1, 2)
    plot2 = ax2.pcolor(samples_x.detach().numpy(), samples_y.detach().numpy(),
                       dV.detach().numpy())
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("V(x[n+1])-V(x[n])")
    fig.colorbar(plot2, ax=ax2)
    plt.show()


def validate_relu(relu, system, state_samples):
    with torch.no_grad():
        for state in state_samples:
            for i in range(system.num_modes):
                if torch.all(system.P[i] @ state <= system.q[i]):
                    next_state = system.A[i] @ state + system.g[i]
                    break
            v_state = relu.forward(state)
            v_next = relu.forward(next_state)
            assert(v_next <= v_state)


if __name__ == "__main__":
    system = test_hybrid_linear_system.setup_trecate_discrete_time_system()
    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(system)

    relu = test_train_lyapunov.setup_relu()

    state_samples_all1 = test_train_lyapunov.setup_state_samples_all((51, 51))
    # First train a ReLU to approximates the value function.
    options1 = train_lyapunov.TrainValueApproximatorOptions()
    options1.max_epochs = 200
    options1.convergence_tolerance = 0.01
    result1 = train_lyapunov.train_value_approximator(
        system, relu, lambda x: x@x, state_samples_all1, options1)
    plot_relu(relu, system, (51, 51))

    x_equilibrium = torch.tensor([0, 0], dtype=torch.float64)
    # Ignore the samples that are close to the origin. These samples don't have
    # to be strictly negative, but can be close to 0.
    state_samples_all = test_train_lyapunov.setup_state_samples_all((21, 21))
    state_samples_all = [sample for sample in state_samples_all if
                         torch.all(torch.abs(sample) <= 0.05)]

    options = train_lyapunov.LyapunovReluTrainingOptions()
    options.output_flag = True
    options.max_iterations = 3000
    options.sample_lyapunov_loss_margin = 0.1
    options.dV_epsilon = 0.

    result = train_lyapunov.train_lyapunov_relu(
        lyapunov_hybrid_system, relu, x_equilibrium, state_samples_all,
        options)

    plot_relu(relu, system, (51, 51))

    assert(result)

    state_samples_validate = test_train_lyapunov.setup_state_samples_all(
        (100, 100))
    validate_relu(relu, system, state_samples_validate)
