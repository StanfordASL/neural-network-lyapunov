import neural_network_lyapunov.examples.car.dubins_car as dubins_car
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import argparse
import torch
import numpy as np
import os


def generate_dynamics_data(dt):
    """
    Generate many pairs of state/action to next state.
    Notice that since the car dynamics is shift invariant, our input doesn't
    include the car position.
    """
    dtype = torch.float64
    plant = dubins_car.DubinsCar(dtype)
    theta_grid = torch.linspace(-2. * np.pi, 2. * np.pi, 301, dtype=dtype)
    vel_grid = torch.linspace(-3, 6, 201, dtype=dtype)
    thetadot_grid = torch.linspace(-0.25 * np.pi,
                                   0.25 * np.pi,
                                   51,
                                   dtype=dtype)

    state_action_data = []
    state_next_data = []

    for i in range(theta_grid.numel()):
        for j in range(vel_grid.numel()):
            for k in range(thetadot_grid.numel()):
                xu = torch.tensor(
                    [0, 0, theta_grid[i], vel_grid[j], thetadot_grid[k]],
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
            network_input_zero = torch.cat(
                (state_action[:, 2].reshape((-1, 1)),
                 torch.zeros((state_action.shape[0], 1),
                             dtype=torch.float64), state_action[:, 4].reshape(
                                 (-1, 1))),
                dim=1)
        else:
            network_input = state_action[:, [2, 3]]
            network_input_zero = torch.cat(
                (state_action[:, 2].reshape((-1, 1)),
                 torch.zeros((state_action.shape[0], 1), dtype=torch.float64)),
                dim=1)
        return model(network_input) - model(network_input_zero)

    utils.train_approximator(training_dataset,
                             dynamics_relu,
                             model_output,
                             batch_size=30,
                             num_epochs=num_epochs,
                             lr=0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dubin's car training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument("--load_dynamics_data",
                        type=str,
                        default=None,
                        help="path to load the dynamics data.")
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--load_controller_relu",
                        type=str,
                        default=None,
                        help="path to load the controller relu.")
    parser.add_argument("--load_lyapunov_relu",
                        type=str,
                        default=None,
                        help="path to load lyapunov relu")
    parser.add_argument("--search_R", action="store_true")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=2000,
        help="max number of iterations in training the controller.")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--train_on_samples", action="store_true")
    parser.add_argument("--pretrain_num_epochs", type=int, default=100)
    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    dt = 0.01
    if args.generate_dynamics_data:
        dynamics_dataset = generate_dynamics_data(dt)

    if args.load_dynamics_data:
        dynamics_dataset = torch.load(args.load_dynamics_data)

    thetadot_as_input = True
    dynamics_relu = utils.setup_relu((3, 8, 8, 2),
                                     params=None,
                                     negative_slope=0.1,
                                     bias=True,
                                     dtype=torch.float64)
    if args.train_forward_model:
        train_forward_model(dynamics_relu,
                            dynamics_dataset,
                            num_epochs=100,
                            thetadot_as_input=thetadot_as_input)
    else:
        dynamics_model_data = torch.load(dir_path +
                                         "/data/dubins_car_forward_relu8.pt")
        dynamics_relu = utils.setup_relu(
            dynamics_model_data["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["negative_slope"],
            bias=dynamics_model_data["bias"],
            dtype=torch.float64)
        dynamics_relu.load_state_dict(dynamics_model_data["state_dict"])

    V_lambda = 0.5
    controller_relu = utils.setup_relu((3, 15, 15, 2),
                                       params=None,
                                       negative_slope=0.1,
                                       bias=True,
                                       dtype=torch.float64)
    if args.load_controller_relu:
        controller_data = torch.load(args.load_controller_relu)
        controller_relu = utils.setup_relu(
            controller_data["linear_layer_width"],
            params=None,
            negative_slope=controller_data["negative_slope"],
            bias=controller_data["bias"],
            dtype=torch.float64)
        controller_relu.load_state_dict(controller_data["state_dict"])

    lyapunov_relu = utils.setup_relu((3, 15, 15, 8, 1),
                                     params=None,
                                     negative_slope=0.1,
                                     bias=True,
                                     dtype=torch.float64)
    R = torch.cat((torch.eye(3, dtype=torch.float64),
                   torch.tensor([[1, -1, 1], [-1, -1, 1], [1, 0, 1], [0.5, -1, 1]],
                                dtype=torch.float64)),
                  dim=0)

    if args.load_lyapunov_relu:
        lyapunov_data = torch.load(args.load_lyapunov_relu)
        lyapunov_relu = utils.setup_relu(
            lyapunov_data["linear_layer_width"],
            params=None,
            negative_slope=lyapunov_data["negative_slope"],
            bias=lyapunov_data["bias"],
            dtype=torch.float64)
        lyapunov_relu.load_state_dict(lyapunov_data["state_dict"])
        V_lambda = lyapunov_data["V_lambda"]
        R = lyapunov_data["R"]

    x_lo = torch.tensor([-0.1, -0.1, -np.pi / 6], dtype=torch.float64)
    x_up = torch.tensor([0.1, 0.1, np.pi / 6], dtype=torch.float64)
    u_lo = torch.tensor([-2, -0.2 * np.pi], dtype=torch.float64)
    u_up = torch.tensor([5, 0.2 * np.pi], dtype=torch.float64)
    forward_system = dubins_car.DubinsCarReLUModel(torch.float64, x_lo, x_up,
                                                   u_lo, u_up, dynamics_relu,
                                                   dt, thetadot_as_input)
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy())
    lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(closed_loop_system,
                                                     lyapunov_relu)

    if args.search_R:
        R_options = train_lyapunov.SearchROptions(R.shape, epsilon=0.5)
        R_options.set_variable_value(R.detach().numpy())
    else:
        R_options = train_lyapunov.FixedROptions(R)

    dut = train_lyapunov.TrainLyapunovReLU(lyap, V_lambda,
                                           closed_loop_system.x_equilibrium,
                                           R_options)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-6
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.4
    dut.lyapunov_derivative_epsilon = 0.001
    dut.learning_rate = 0.001
    state_samples_all = utils.get_meshgrid_samples(x_lo,
                                                   x_up, (31, 31, 31),
                                                   dtype=torch.float64)
    dut.output_flag = True
    if args.train_on_samples:
        dut.train_lyapunov_on_samples(state_samples_all,
                                      num_epochs=args.pretrain_num_epochs,
                                      batch_size=50)
    dut.enable_wandb = args.enable_wandb
    dut.add_derivative_adversarial_state = True
    dut.lyapunov_derivative_sample_cost_weight = 5.
    dut.train(torch.empty((0, 3), dtype=torch.float64))
    pass
