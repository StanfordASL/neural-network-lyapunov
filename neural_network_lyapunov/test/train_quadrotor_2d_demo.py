import neural_network_lyapunov.test.quadrotor_2d as quadrotor_2d
import neural_network_lyapunov.utils as utils

import torch
import numpy as np
import scipy.integrate
import argparse
import os


def generate_quadrotor_dynamics_data(dt):
    """
    Generate the pairs (x[n], u[n]) -> (x[n+1])
    """
    dtype = torch.float64
    plant = quadrotor_2d.Quadrotor2D(dtype)

    theta_range = [-np.pi / 3, np.pi / 3]
    ydot_range = [-5, 5]
    zdot_range = [-5, 5]
    thetadot_range = [-2, 2]
    u_range = [-5, 15]

    xu_tensors = []
    x_next_tensors = []

    def scale_uniform_samples(samples, row, lo, up):
        # scale uniform samples[row, :] to the range [lo, up]
        samples[row, :] *= (up - lo)
        samples[row, :] += lo
    # We don't need to take the grid on y and z dimension of the quadrotor,
    # since the dynamics is invariant along these dimensions.
    x_samples = torch.cat((torch.zeros(2, 1000, dtype=torch.float64),
                           torch.rand(4, 1000, dtype=torch.float64)), dim=0)
    u_samples = torch.rand(2, 1000, dtype=torch.float64)
    scale_uniform_samples(x_samples, 2, theta_range[0], theta_range[1])
    scale_uniform_samples(x_samples, 3, ydot_range[0], ydot_range[1])
    scale_uniform_samples(x_samples, 4, zdot_range[0], zdot_range[1])
    scale_uniform_samples(x_samples, 5, thetadot_range[0], thetadot_range[1])
    scale_uniform_samples(u_samples, 0, u_range[0], u_range[1])
    scale_uniform_samples(u_samples, 1, u_range[0], u_range[1])
    for i in range(x_samples.shape[1]):
        for j in range(u_samples.shape[1]):
            result = scipy.integrate.solve_ivp(lambda t, x: plant.dynamics(
                x, u_samples[:, j].detach().numpy()), (0, dt),
                x_samples[:, i].detach().numpy())
            xu_tensors.append(
                torch.cat((x_samples[:, i], u_samples[:, j])).reshape((1, -1)))
            x_next_tensors.append(torch.from_numpy(
                result.y[:, -1]).reshape((1, -1)))
    dataset_input = torch.cat(xu_tensors, dim=0)
    dataset_output = torch.cat(x_next_tensors, dim=0)
    return torch.utils.data.TensorDataset(dataset_input, dataset_output)


def train_forward_model(forward_model, model_dataset):
    # The forward model maps (theta[n], u1[n], u2[n]) to
    # (ydot[n+1]-ydot[n], zdot[n+1]-zdot[n], thetadot[n+1]-thetadot[n])
    plant = quadrotor_2d.Quadrotor2D(torch.float64)
    u_equilibrium = torch.full(
        (2,), plant.mass * plant.gravity / 2, dtype=torch.float64)

    xu_inputs, x_next_outputs = model_dataset[:]
    network_input_data = xu_inputs[:, [2, 6, 7]]
    network_output_data = x_next_outputs[:, 3:] - xu_inputs[:, 3:6]
    v_dataset = torch.utils.data.TensorDataset(
        network_input_data, network_output_data)

    def compute_next_v(model, theta_u):
        return model(theta_u) - model(torch.cat(
            (torch.tensor([0], dtype=torch.float64), u_equilibrium)))
    utils.train_approximator(
        v_dataset, forward_model, compute_next_v, batch_size=50,
        num_epochs=100, lr=0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="quadrotor 2d training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument("--load_dynamics_data", type=str, default=None,
                        help="path to the dynamics data.")
    parser.add_argument("--train_forward_model", action="store_true")
    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dt = 0.01
    if args.generate_dynamics_data:
        model_dataset = generate_quadrotor_dynamics_data(dt)

    if args.load_dynamics_data is not None:
        model_dataset = torch.load(args.load_dynamics_data)

    forward_model = utils.setup_relu(
        (3, 6, 6, 3), params=None, bias=True, negative_slope=0.01,
        dtype=torch.float64)
    if args.train_forward_model:
        train_forward_model(forward_model, model_dataset)
