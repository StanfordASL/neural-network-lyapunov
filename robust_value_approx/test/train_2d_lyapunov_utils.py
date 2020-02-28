import torch

import numpy as np

import robust_value_approx.lyapunov as lyapunov

import matplotlib.pyplot as plt
from matplotlib import cm # noqa
from mpl_toolkits import mplot3d # noqa
from matplotlib import rc


def setup_state_samples_all(x_equilibrium, lower, upper, mesh_size, theta):
    """
    Generate samples in a rotated box region R(θ) * (x* + box) where box is
    lower <= x <= upper
    """
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
    if isinstance(theta, float):
        theta = torch.tensor(theta, dtype=dtype)
    c_theta = torch.cos(theta)
    s_theta = torch.sin(theta)
    R = torch.tensor([[c_theta, -s_theta], [s_theta, c_theta]], dtype=dtype)
    for i in range(mesh_size[0]):
        for j in range(mesh_size[1]):
            state_samples[i * mesh_size[1] + j] = \
                R @ (x_equilibrium + torch.tensor(
                    [samples_x[i, j], samples_y[i, j]], dtype=dtype))
    return torch.stack(state_samples, dim=0)

def setup_state_samples_on_boundary(x_equilibrium, lower, upper, mesh_size, theta):
    """
    Generate samples one the boundary of a rotated box region R(θ) * (x* + box)
    where box is lower <= x <= upper
    """
    assert(isinstance(lower, torch.Tensor))
    assert(isinstance(upper, torch.Tensor))
    assert(lower.shape == (2,))
    assert(upper.shape == (2,))
    assert(len(mesh_size) == 2)
    dtype = torch.float64
    samples_x = torch.cat([
        torch.linspace(lower[0], upper[0], mesh_size[0]),
        torch.linspace(lower[0], upper[0], mesh_size[0]),
        torch.full((mesh_size[1],), lower[0]),
        torch.full((mesh_size[1],), upper[0])]).type(dtype)
    samples_y = torch.cat([
        torch.full((mesh_size[0],), lower[1]),
        torch.full((mesh_size[1],), upper[1]),
        torch.linspace(lower[1], upper[1], mesh_size[1]),
        torch.linspace(lower[1], upper[1], mesh_size[1])]).type(dtype)
    samples = torch.stack([samples_x, samples_y])
    if isinstance(theta, float):
        theta = torch.tensor(theta, dtype=dtype)
    c_theta = torch.cos(theta)
    s_theta = torch.sin(theta)
    R = torch.tensor([[c_theta, -s_theta], [s_theta, c_theta]], dtype=dtype)
    return (R @ (samples + x_equilibrium.reshape((2, 1)))).T

def plot_relu(
    relu, system, V_rho, lyapunov_positivity_epsilon,
    lyapunov_derivative_epsilon, x_equilibrium, lower, upper, mesh_size,
        theta, vmin=None, vmax=None, discrete_time=True, cmap=cm.coolwarm):
    """
    Draw 3 subplots
    top plot: Lyapunov function in 3D.
    middle plot: V-ε₁*|x-x*|₁ as a colormap.
    bottom plot: V̇ + ε₂*V as a colormap
    where V = nn(x) - nn(x*) + ρ|x-x*|₁
    In a box region defined as R(θ) * (x* + box) where box is
    lower <= x <= upper
    @param relu The ReLU network
    @param system An AutonomousHybridLinearSystem
    @param V_rho ρ in the definition of V.
    @param lyapunov_positivity_epsilon ε₁ defined above.
    @param lyapunov_derivative_epsilon ε₂ defined above.
    @param x_equilibrium x*.
    @param lower, upper defined above.
    @param mesh_size A length 2 tuple. The number of point along each axis in
    the box.
    @boxparam theta The rotation angle θ.
    """
    if discrete_time:
        dut = lyapunov.LyapunovDiscreteTimeHybridSystem(system)
    else:
        dut = lyapunov.LyapunovContinuousTimeHybridSystem(system)
    assert(isinstance(mesh_size, tuple))
    assert(len(mesh_size) == 2)
    dtype = torch.float64
    with torch.no_grad():
        state_samples_all = setup_state_samples_all(
            x_equilibrium, lower, upper, mesh_size, theta)
        samples_x = torch.empty(mesh_size)
        samples_y = torch.empty(mesh_size)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                samples_x[i, j] = state_samples_all[i * mesh_size[1] + j][0]
                samples_y[i, j] = state_samples_all[i * mesh_size[1] + j][1]
        V = torch.zeros(mesh_size)
        V_positivity = torch.zeros(mesh_size)
        dV = torch.zeros(mesh_size)
        relu_at_equilibrium = relu.forward(x_equilibrium)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                state_sample = torch.tensor(
                    [samples_x[i, j], samples_y[i, j]], dtype=dtype)
                V[i, j] = dut.lyapunov_value(
                    relu, state_sample, x_equilibrium, V_rho,
                    relu_at_equilibrium)
                V_positivity[i, j] = V[i, j] -\
                    lyapunov_positivity_epsilon * torch.norm(
                        state_sample - x_equilibrium, p=1)
                dV[i, j] = torch.max(torch.cat(dut.lyapunov_derivative(
                    state_sample, relu, x_equilibrium, V_rho,
                    lyapunov_derivative_epsilon)))
        samples_x_np = samples_x.detach().numpy()
        samples_y_np = samples_y.detach().numpy()
        V_np = V.detach().numpy()
        V_positivity_np = V_positivity.detach().numpy()
        dV_np = dV.detach().numpy()
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1, projection='3d')
        ax1.plot_surface(samples_x_np, samples_y_np, V_np, cmap=cm.coolwarm)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("V")

        ax2 = fig.add_subplot(3, 1, 2)
        plot2 = ax2.pcolor(samples_x_np, samples_y_np, V_positivity_np)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        rc('text', usetex=True)
        ax2.set_title(r"$V-\epsilon_1|x-x^*|_1$")
        fig.colorbar(plot2, ax=ax2)

        ax3 = fig.add_subplot(3, 1, 3)
        plot3 = ax3.pcolor(
            samples_x_np, samples_y_np, dV_np, vmin=vmin, vmax=vmax,
            cmap=cmap)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        if discrete_time:
            ax3.set_title(r"$V(x[n+1])-V(x[n])+\epsilon_2V(x[n])")
        else:
            ax3.set_title(r"$\dot{V} + \epsilon_2V$")
        fig.colorbar(plot3, ax=ax3)
        plt.show()


def plot_cost_to_go_approximator(
    relu, x0_value_samples, x_equilibrium, V_rho, lower, upper, mesh_size,
        theta):
    """
    Plot the cost-to-go approximator and the cost-to-go samples.
    The approximator is nn(x) - nn(x*) + ρ|x-x*|₁
    """
    dtype = torch.float64
    with torch.no_grad():
        state_samples_all = setup_state_samples_all(
            x_equilibrium, lower, upper, mesh_size, theta)
        samples_x = torch.empty(mesh_size)
        samples_y = torch.empty(mesh_size)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                samples_x[i, j] = state_samples_all[i * mesh_size[1] + j][0]
                samples_y[i, j] = state_samples_all[i * mesh_size[1] + j][1]
        V = torch.zeros(mesh_size)
        relu_at_equilibrium = relu.forward(x_equilibrium)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                state_sample = torch.tensor(
                    [samples_x[i, j], samples_y[i, j]], dtype=dtype)
                V[i, j] = relu(state_sample) - relu_at_equilibrium + \
                    V_rho * torch.norm(state_sample - x_equilibrium, p=1)

        samples_x_np = samples_x.detach().numpy()
        samples_y_np = samples_y.detach().numpy()
        V_np = V.detach().numpy()
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.plot_surface(samples_x_np, samples_y_np, V_np, cmap=cm.coolwarm)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("V")

        theta_tensor = torch.tensor(theta, dtype=dtype)
        c_theta = torch.cos(theta_tensor)
        s_theta = torch.sin(theta_tensor)
        R = torch.tensor(
            [[c_theta, -s_theta], [s_theta, c_theta]], dtype=dtype)

        def is_point_in_box(pt):
            pt_in_box = R.T @ (pt - x_equilibrium)
            return torch.all(pt_in_box >= lower) and\
                torch.all(pt_in_box <= upper)
        x0_value_samples_np = np.array(
            [[pair[0][0], pair[0][1], pair[1]] for pair in x0_value_samples
             if is_point_in_box(pair[0])])
        ax1.plot(
            x0_value_samples_np[:, 0], x0_value_samples_np[:, 1],
            x0_value_samples_np[:, 2], linestyle='None', marker='.',
            markersize=1.)

        plt.show()
