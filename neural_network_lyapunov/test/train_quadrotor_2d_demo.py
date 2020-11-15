import neural_network_lyapunov.test.quadrotor_2d as quadrotor_2d

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
    x_samples = torch.cat((torch.zeros(2, 1000), torch.rand(4, 1000)), dim=0)
    u_samples = torch.rand(2, 1000)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="quadrotor 2d training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument("--train_forward_model", action="store_true")
    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dt = 0.01
    if args.generate_dynamics_data:
        model_dataset = generate_quadrotor_dynamics_data(dt)


