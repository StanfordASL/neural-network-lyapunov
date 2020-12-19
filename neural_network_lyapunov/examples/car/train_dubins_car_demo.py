import neural_network_lyapunov.examples.car.dubins_car as dubins_car
import neural_network_lyapunov.utils as utils
import argparse
import torch
import numpy as np


def generate_dynamics_data():
    """
    Generate many pairs of (theta, vel) to (delta_pos_x, delta_pos_y)
    Notice that since the car dynamics is shift invariant, our input doesn't
    include the car position.
    """
    dtype = torch.float64
    plant = dubins_car.DubinsCar(dtype)
    theta_grid = torch.linspace(-2.5 * np.pi, 2.5 * np.pi, 401, dtype=dtype)
    vel_grid = torch.linspace(-4, 8, 201, dtype=dtype)

    input_data = []
    output_data = []
    dt = 0.01

    for i in range(theta_grid.numel()):
        for j in range(vel_grid.numel()):
            input_data.append(torch.tensor([[theta_grid[i], vel_grid[j]]]))
            x_next = plant.next_pose(torch.tensor(
                [0, 0, theta_grid[i]], dtype=dtype),
                torch.tensor([vel_grid[j], 0], dtype=dtype), dt)
            output_data.append(
                torch.tensor([[x_next[0], x_next[1]]], dtype=dtype))
    input_data = torch.cat(input_data, dim=0)
    output_data = torch.cat(output_data, dim=0)
    return torch.utils.data.TensorDataset(input_data, output_data)


def train_forward_model(dynamics_relu, dataset, num_epochs):
    """
    The dataset contains the mapping from (theta, vel) to
    (delta_pos_x, delta_pos_y)
    """

    def model_output(model, model_input):
        return model(model_input)

    utils.train_approximator(
        dataset, dynamics_relu, model_output, batch_size=30,
        num_epochs=num_epochs, lr=0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dubin's car training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument("--load_dynamics_data", type=str, default=None,
                        help="path to load the dynamics data.")
    parser.add_argument("--train_forward_model", action="store_true")
    args = parser.parse_args()

    if args.generate_dynamics_data:
        dynamics_dataset = generate_dynamics_data()

    if args.load_dynamics_data:
        dynamics_dataset = torch.load(args.load_dynamics_data)

    dynamics_relu = utils.setup_relu(
        (2, 4, 4, 2), params=None, negative_slope=0.1, bias=True,
        dtype=torch.float64)
    if args.train_forward_model:
        train_forward_model(dynamics_relu, dynamics_dataset, num_epochs=100)
    pass
