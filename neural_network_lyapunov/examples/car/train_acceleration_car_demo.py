"""
Do not run this code!!!

Dubins car cannot be stabilized by a continuous controller.
"""
import neural_network_lyapunov.examples.car.acceleration_car as\
    acceleration_car
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.train_lyapunov_barrier as train_lyapunov_barrier
import neural_network_lyapunov.r_options as r_options
import argparse
import torch
import numpy as np
import os
import scipy.integrate


def generate_dynamics_data(dt):
    dtype = torch.float64
    plant = acceleration_car.AccelerationCar(dtype)

    theta_vel_lo = torch.tensor([-1.2 * np.pi, -2], dtype=dtype)
    theta_vel_up = torch.tensor([1.2 * np.pi, 5], dtype=dtype)
    u_lo = torch.tensor([-np.pi / 6, -4], dtype=dtype)
    u_up = torch.tensor([np.pi / 6, 4], dtype=dtype)

    x_samples = utils.uniform_sample_in_box(theta_vel_lo, theta_vel_up, 3000)
    x_samples = torch.cat((torch.zeros(
        (x_samples.shape[0], 2), dtype=dtype), x_samples),
                          dim=1)
    u_samples = utils.uniform_sample_in_box(u_lo, u_up, 3000)

    xu_data = []
    x_next_data = []
    for i in range(x_samples.shape[0]):
        for j in range(u_samples.shape[0]):
            result = scipy.integrate.solve_ivp(
                lambda t, x: plant.dynamics(x, u_samples[j].detach().numpy()),
                (0, dt), x_samples[i].detach().numpy())
            xu_data.append(
                torch.cat((x_samples[i], u_samples[j])).reshape((1, -1)))
            x_next_data.append(
                torch.from_numpy(result.y[:, -1]).reshape((1, -1)))

    xu_data = torch.cat(xu_data, dim=0)
    x_next_data = torch.cat(x_next_data, dim=0)
    return torch.utils.data.TensorDataset(xu_data, x_next_data)


def train_forward_model(forward_model, model_dataset, num_epochs):
    """
    The forward model network maps (theta[n], vel[n], theta_dot[n], accel[n])
    to pos[n+1] - pos[n]
    """
    xu_inputs, x_next_outputs = model_dataset[:]
    network_input_data = xu_inputs[:, [2, 3, 4, 5]]
    network_output_data = x_next_outputs[:, :2] - xu_inputs[:, :2]
    training_dataset = torch.utils.data.TensorDataset(network_input_data,
                                                      network_output_data)

    def compute_delta_pos(model, network_input):
        return model(network_input) - model(
            torch.zeros((4, ), dtype=torch.float64))

    utils.train_approximator(training_dataset,
                             forward_model,
                             compute_delta_pos,
                             batch_size=50,
                             num_epochs=num_epochs,
                             lr=0.001)


if __name__ == "__main__":
    raise Exception("Dubins car cannot be stabilized by a continuous controller, refer" +
                    "to https://arxiv.org/pdf/math/9902026.pdf for a proof. Using" +
                    "neural-network controllers is doomed to fail for Dubins car.")
    parser = argparse.ArgumentParser(
        description="Acceleration car training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument("--load_dynamics_data",
                        type=str,
                        default=None,
                        help="path to the forward dynamics data.")
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--load_forward_model",
                        type=str,
                        default=None,
                        help="path to the forward model")
    parser.add_argument("--load_lyapunov_relu",
                        type=str,
                        default=None,
                        help="path to the lyapunov relu.")
    parser.add_argument("--load_controller_relu",
                        type=str,
                        default=None,
                        help="path to the controller relu.")
    parser.add_argument("--search_R", action="store_true")
    parser.add_argument("--train_on_samples", action="store_true")
    parser.add_argument("--pretrain_num_epochs",
                        type=int,
                        default=10,
                        help="number of epochs when pretrain on samples.")
    parser.add_argument("--max_iterations",
                        type=int,
                        default=5000,
                        help="max iterations in training Lyapunov.")
    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    dtype = torch.float64
    dt = 0.01

    if args.generate_dynamics_data:
        dynamics_dataset = generate_dynamics_data(dt)

    if args.load_dynamics_data:
        dynamics_dataset = torch.load(args.load_dynamics_data)

    if args.train_forward_model:
        forward_model = utils.setup_relu((4, 8, 8, 2),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=dtype)
        train_forward_model(forward_model, dynamics_dataset, num_epochs=100)
    elif args.load_forward_model:
        forward_model_data = torch.load(args.load_forward_model)
        forward_model = utils.setup_relu(
            forward_model_data["linear_layer_width"],
            params=None,
            bias=forward_model_data["bias"],
            negative_slope=forward_model_data["negative_slope"],
            dtype=dtype)
        forward_model.load_state_dict(forward_model_data["state_dict"])

        R = torch.cat(
            (torch.eye(4, dtype=dtype),
             torch.tensor([[1., 0., 1., 0.], [-1, 1., 0., 1.]], dtype=dtype)),
            dim=0)

        lyapunov_relu = utils.setup_relu((4, 8, 8, 6, 1),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=dtype)
        V_lambda = 0.8
    if args.load_lyapunov_relu is not None:
        lyapunov_data = torch.load(args.load_lyapunov_relu)
        lyapunov_relu = utils.setup_relu(
            lyapunov_data["linear_layer_width"],
            params=None,
            negative_slope=lyapunov_data["negative_slope"],
            bias=lyapunov_data["bias"],
            dtype=dtype)
        lyapunov_relu.load_state_dict(lyapunov_data["state_dict"])
        V_lambda = lyapunov_data["V_lambda"]
        R = lyapunov_data["R"]

    controller_relu = utils.setup_relu((4, 6, 4, 2),
                                       params=None,
                                       negative_slope=0.1,
                                       bias=True,
                                       dtype=dtype)
    if args.load_controller_relu is not None:
        controller_data = torch.load(args.load_controller_relu)
        controller_relu = utils.setup_relu(
            controller_data["linear_layer_width"],
            params=None,
            negative_slope=controller_data["negative_slope"],
            bias=controller_data["bias"],
            dtype=dtype)
        controller_relu.load_state_dict(controller_data["state_dict"])

    x_lo = torch.tensor([-0.5, -0.5, -np.pi * 0.3, -1], dtype=dtype)
    x_up = torch.tensor([0.5, 0.5, np.pi * 0.3, 2], dtype=dtype)
    u_lo = torch.tensor([-np.pi / 6, -4], dtype=dtype)
    u_up = torch.tensor([np.pi / 6, 4], dtype=dtype)
    forward_system = acceleration_car.AccelerationCarReLUModel(
        dtype, x_lo, x_up, u_lo, u_up, forward_model, dt)
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy())
    lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(closed_loop_system,
                                                     lyapunov_relu)

    if args.search_R:
        R_options = r_options.SearchRwithSPDOptions(R.shape, 0.1)
        R_options.set_variable_value(R.detach().numpy())
    else:
        R_options = r_options.FixedROptions(R)
    dut = train_lyapunov_barrier.Trainer()
    dut.add_lyapunov(lyap, V_lambda, closed_loop_system.x_equilibrium,
                     R_options)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-5
    dut.lyapunov_positivity_convergence_tol = 5e-6
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.4
    dut.lyapunov_derivative_epsilon = 0.004
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    state_samples_all = utils.get_meshgrid_samples(x_lo,
                                                   x_up, (5, 5, 5, 5),
                                                   dtype=dtype)
    dut.output_flag = True
    if args.train_on_samples:
        dut.train_lyapunov_on_samples(state_samples_all,
                                      num_epochs=args.pretrain_num_epochs,
                                      batch_size=50)
    dut.enable_wandb = True
    dut.train(torch.empty((0, 4), dtype=dtype))
    pass
