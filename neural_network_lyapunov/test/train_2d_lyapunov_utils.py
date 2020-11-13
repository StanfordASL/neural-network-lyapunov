import torch

import numpy as np

import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization

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
        torch.linspace(lower[0] - x_equilibrium[0],
                       upper[0] - x_equilibrium[0], mesh_size[0], dtype=dtype),
        torch.linspace(lower[1] - x_equilibrium[1],
                       upper[1] - x_equilibrium[1], mesh_size[1], dtype=dtype))
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


def setup_state_samples_on_boundary(
        x_equilibrium, lower, upper, mesh_size, theta):
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


def plot_relu_domain_boundary(ax, relu, **kwargs):
    """
    For a ReLU network with no bias term, draw the boundary between each linear
    piece in the network piecewise linear output.
    """
    for layer in relu:
        if isinstance(layer, torch.nn.Linear):
            assert(layer.bias is None)
    dtype = torch.float64
    theta = np.linspace(0, 2 * np.pi, 500)
    rays = []
    P = None
    with torch.no_grad():
        for i in range(theta.shape[0] - 1):
            x = torch.tensor([np.cos(theta[i]), np.sin(theta[i])], dtype=dtype)
            if i != 0 and torch.all(P @ x <= 0):
                continue
            activation_pattern = relu_to_optimization.\
                ComputeReLUActivationPattern(relu, x)
            _, _, P, _ = relu_to_optimization.ReLUGivenActivationPattern(
                relu, 2, activation_pattern, dtype)
            # Now draw the boundary of P*x<=0
            for j in range(P.shape[0]):
                ray = torch.tensor([P[j, 1], -P[j, 0]], dtype=dtype)
                ray_magnitude = torch.norm(ray, p=2)
                if torch.all(P @ ray <= 1e-12):
                    ax.plot(
                        [0, ray[0] / ray_magnitude * 10],
                        [0, ray[1] / ray_magnitude * 10], **kwargs)
                    rays.append(ray)
                ray = -ray
                if torch.all(P @ ray <= 1e-12):
                    ax.plot(
                        [0, ray[0] / ray_magnitude * 10],
                        [0, ray[1] / ray_magnitude * 10], **kwargs)
                    rays.append(ray)
    return rays


def plot_lyapunov(
    ax, relu, V_lambda, x_equilibrium, R, x_lower, x_upper, mesh_size,
        fontsize, **kwargs):
    """
    Plot V(x) in 3D.
    where V(x) = relu(x) + ρ|x-x*|₁
    """
    with torch.no_grad():
        state_samples_all = setup_state_samples_all(
            x_equilibrium, x_lower, x_upper, mesh_size, 0.)
        V = relu(state_samples_all) - relu(x_equilibrium)\
            + V_lambda * torch.norm(
                R @ (state_samples_all - x_equilibrium).T, p=1, dim=0
                ).reshape((-1, 1))
        V_np = np.empty(mesh_size)
        samples_x = torch.empty(mesh_size)
        samples_y = torch.empty(mesh_size)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                samples_x[i, j] = state_samples_all[i * mesh_size[1] + j][0]
                samples_y[i, j] = state_samples_all[i * mesh_size[1] + j][1]
                V_np[i, j] = V[i * mesh_size[1] + j].item()
        samples_x_np = samples_x.detach().numpy()
        samples_y_np = samples_y.detach().numpy()
        ax.plot_surface(samples_x_np, samples_y_np, V_np, **kwargs)
        ax.set_xlabel("x(1)", fontsize=fontsize)
        ax.set_ylabel("x(2)", fontsize=fontsize)
        ax.set_zlabel("V", fontsize=fontsize)


def plot_lyapunov_colormap(
    fig, ax, relu, V_lambda, lyapunov_positivity_epsilon, x_equilibrium, R,
        x_lower, x_upper, mesh_size, title_fontsize, **kwargs):
    """
    Plot V(x) - epsilon |x - x*|₁ as a color map.
    where V(x) = relu(x) + λ|x-x*|₁
    """
    with torch.no_grad():
        state_samples_all = setup_state_samples_all(
            x_equilibrium, x_lower, x_upper, mesh_size, 0.)
        V = relu(state_samples_all) - relu(x_equilibrium) +\
            V_lambda * torch.norm(
                R @ (state_samples_all - x_equilibrium).T, p=1, dim=0
                ).reshape((-1, 1))
        V_minus_l1 = V - lyapunov_positivity_epsilon * torch.norm(
            R @ (state_samples_all - x_equilibrium), p=1, dim=0).reshape(
                (-1, 1))
        V_minus_l1_np = np.empty(mesh_size)
        samples_x = torch.empty(mesh_size)
        samples_y = torch.empty(mesh_size)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                samples_x[i, j] = state_samples_all[i * mesh_size[1] + j][0]
                samples_y[i, j] = state_samples_all[i * mesh_size[1] + j][1]
                V_minus_l1_np[i, j] = V_minus_l1[i * mesh_size[1] + j].item()
        samples_x_np = samples_x.detach().numpy()
        samples_y_np = samples_y.detach().numpy()
        plot = ax.pcolor(
            samples_x_np, samples_y_np, V_minus_l1_np, **kwargs)
        rc('text', usetex=True)
        if lyapunov_positivity_epsilon == 0:
            ax.set_title(r"$V$", fontsize=title_fontsize)
        else:
            ax.set_title(r"$V-\epsilon_1|x|_1$", fontsize=title_fontsize)
        ax.set_xlabel("x(1)", fontsize=title_fontsize)
        ax.set_ylabel("x(2)", fontsize=title_fontsize)
        cb = fig.colorbar(plot, ax=ax)
        cb.ax.tick_params(labelsize=title_fontsize)


def plot_lyapunov_dot_colormap(
    fig, ax, relu, system, V_lambda, lyapunov_derivative_epsilon,
    x_equilibrium, x_lower, x_upper, mesh_size, discrete_time, fontsize,
        **kwargs):
    if discrete_time:
        dut = lyapunov.LyapunovDiscreteTimeHybridSystem(system, relu)
    else:
        dut = lyapunov.LyapunovContinuousTimeHybridSystem(system, relu)
    dtype = torch.float64
    with torch.no_grad():
        state_samples_all = setup_state_samples_all(
            x_equilibrium, x_lower, x_upper, mesh_size, 0.)
        samples_x = torch.empty(mesh_size)
        samples_y = torch.empty(mesh_size)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                samples_x[i, j] = state_samples_all[i * mesh_size[1] + j][0]
                samples_y[i, j] = state_samples_all[i * mesh_size[1] + j][1]
        dV = torch.zeros(mesh_size)
        for i in range(mesh_size[0]):
            for j in range(mesh_size[1]):
                state_sample = torch.tensor(
                    [samples_x[i, j], samples_y[i, j]], dtype=dtype)
                dV[i, j] = torch.max(torch.cat(dut.lyapunov_derivative(
                    state_sample, x_equilibrium, V_lambda,
                    lyapunov_derivative_epsilon)))
        samples_x_np = samples_x.detach().numpy()
        samples_y_np = samples_y.detach().numpy()
        dV_np = dV.detach().numpy()

        plot = ax.pcolor(
            samples_x_np, samples_y_np, dV_np, **kwargs)
        ax.set_xlabel("x(1)", fontsize=fontsize)
        ax.set_ylabel("x(2)", fontsize=fontsize)
        if discrete_time:
            ax.set_title(
                r"$V(x_{n+1})-V(x_n)+\epsilon_2V(x_n)", fontsize=fontsize)
        else:
            ax.set_title(r"$\dot{V} + \epsilon_2V$", fontsize=fontsize)
        cb = fig.colorbar(plot, ax=ax)
        cb.ax.tick_params(labelsize=fontsize)


def plot_sublevel_set(
        ax, relu, V_lambda, x_equilibrium, R, upper_bound, **kwargs):
    """
    Plot the sub-level set of V(x) <= upper_bound where
    V(x) = network(x) + λ |x-x*|₁
    currently we only accept network with linear and (leaky) ReLU units. The
    linear unit cannot have bias term.
    """
    for layer in relu:
        if isinstance(layer, torch.nn.Linear):
            assert(layer.bias is None)
    # Shoot a ray with angle theta. Along the ray, the Lyapunov function is a
    # linear function of the ray length.
    dtype = x_equilibrium.dtype
    N = 500
    theta = torch.linspace(0, 2*np.pi, N).type(dtype)
    x = torch.empty((N, 2), dtype=dtype)
    x[:, 0] = torch.cos(theta)
    x[:, 1] = torch.sin(theta)
    lyapunov_val = relu(x).squeeze() + \
        V_lambda * torch.norm(R @ (x - x_equilibrium).T, p=1, dim=0)
    x = x * (((upper_bound / lyapunov_val).unsqueeze(1)).repeat([1, 2]))
    ax.plot(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), **kwargs)
    return x


def plot_phase_portrait(
        ax, system, x_lower, x_upper, mesh_size, fontsize, **kwargs):
    """
    Plots the phase portrain for a 2D continuous time system.
    @param ax matplotlib axes
    @param system An AutonomousHybridLinearSystem as a continuous time system.
    @param x_lower The lower left corner of the region.
    @param x_upper The upper right corner of the region.
    @param mesh_size A size 2 tuple. Must be pure imaginary number.
    """
    assert(isinstance(
        system, hybrid_linear_system.AutonomousHybridLinearSystem))
    assert(isinstance(x_lower, torch.Tensor))
    assert(isinstance(x_upper, torch.Tensor))
    assert(x_lower.shape == (2,))
    assert(x_upper.shape == (2,))
    assert(len(mesh_size) == 2)
    assert(mesh_size[0] - np.real(mesh_size[0]) == mesh_size[0])
    assert(mesh_size[1] - np.real(mesh_size[1]) == mesh_size[1])
    X1, X2 = np.mgrid[x_lower[0].item():x_upper[0].item():mesh_size[0],
                      x_lower[1].item():x_upper[1].item():mesh_size[1]]
    xdot1 = np.empty(X1.shape)
    xdot2 = np.empty(X2.shape)
    with torch.no_grad():
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                xdot = system.step_forward(
                    torch.tensor([X1[i, j], X2[i, j]], dtype=system.dtype))
                xdot1[i, j] = xdot[0]
                xdot2[i, j] = xdot[1]

    ax.set_xlabel("x(1)", fontsize=fontsize)
    ax.set_ylabel("x(2)", fontsize=fontsize)
    strm = ax.streamplot(X1.T[0], X2[0], xdot1.T, xdot2.T,  **kwargs)
    return strm


def plot_cost_to_go_approximator(
    relu, x0_value_samples, x_equilibrium, V_lambda, R, lower, upper,
        mesh_size, theta):
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
                    V_lambda * torch.norm(
                        R @ (state_sample - x_equilibrium), p=1)

        samples_x_np = samples_x.detach().numpy()
        samples_y_np = samples_y.detach().numpy()
        V_np = V.detach().numpy()
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.plot_surface(samples_x_np, samples_y_np, V_np, cmap=cm.coolwarm)
        ax1.set_xlabel("x(1)")
        ax1.set_ylabel("x(2)")
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
