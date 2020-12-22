import neural_network_lyapunov.examples.car.dubins_car as dubins_car
import neural_network_lyapunov.utils as utils
import argparse
import torch
import numpy as np


def generate_dynamics_data():
    """
    Generate many pairs of state/action to next state.
    Notice that since the car dynamics is shift invariant, our input doesn't
    include the car position.
    """
    dtype = torch.float64
    plant = dubins_car.DubinsCar(dtype)
    theta_grid = torch.linspace(-2.5 * np.pi, 2.5 * np.pi, 201, dtype=dtype)
    vel_grid = torch.linspace(-4, 8, 201, dtype=dtype)
    thetadot_grid = torch.linspace(-0.3 * np.pi, 0.3 * np.pi, 51, dtype=dtype)

    state_action_data = []
    state_next_data = []
    dt = 0.01

    for i in range(theta_grid.numel()):
        for j in range(vel_grid.numel()):
            for k in range(thetadot_grid.numel()):
                xu = torch.tensor([
                    0, 0, theta_grid[i], vel_grid[j], thetadot_grid[k]],
                    dtype=dtype)
                x_next = plant.next_pose(xu[:3], xu[3:], dt)
                state_action_data.append(xu.reshape((1, -1)))
                state_next_data.append(
                    torch.from_numpy(x_next).reshape((1, -1)))
    state_action_data = torch.cat(state_action_data, dim=0)
    state_next_data = torch.cat(state_next_data, dim=0)
    return torch.utils.data.TensorDataset(state_action_data, state_next_data)


def train_forward_model(dynamics_relu, dataset, num_epochs, thetadot_as_input):
    """
    The dataset contains the mapping from state_action to state_next
    """
    (xu_inputs, x_next_outputs) = dataset[:]
    # The network output is just the delta_pos_x and delta_pos_y
    training_dataset = torch.utils.data.TensorDataset(
        xu_inputs, x_next_outputs[:, :2] - xu_inputs[:, :2])

    def model_output(model, state_action):
        # state_action is the concatenation of state (pos_x, pos_y, theta) and
        # the action (vel, thetadot).
        if thetadot_as_input:
            network_input = state_action[:, [2, 3, 4]]
            network_input_zero = torch.zeros((3,), dtype=state_action.dtype)
        else:
            network_input = state_action[:, [2, 3]]
            network_input_zero = torch.zeros((2,), dtype=state_action.dtype)
        return model(network_input) - model(network_input_zero)

    utils.train_approximator(
        training_dataset, dynamics_relu, model_output, batch_size=30,
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

    thetadot_as_input = True
    dynamics_relu = utils.setup_relu(
        (3, 4, 4, 2), params=None, negative_slope=0.1, bias=True,
        dtype=torch.float64)
    if args.train_forward_model:
        train_forward_model(
            dynamics_relu, dynamics_dataset, num_epochs=100,
            thetadot_as_input=thetadot_as_input)
    pass
