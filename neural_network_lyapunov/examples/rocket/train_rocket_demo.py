import neural_network_lyapunov.examples.rocket.rocket as rocket
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.feedback_system as feedback_system

import argparse
import numpy as np
import torch
import scipy.integrate
import scipy.linalg


def generate_dynamics_data(dt):
    dtype = torch.float64
    plant = rocket.Rocket()

    theta_range = [-np.pi / 3, np.pi / 3]
    thetadot_range = [-np.pi, np.pi]
    u0_range = [-plant.hover_thrust * 0.15, plant.hover_thrust * 0.15]
    u1_range = [0, 3 * plant.hover_thrust]
    theta_thetadot_samples = utils.uniform_sample_in_box(
        torch.tensor([theta_range[0], thetadot_range[0]], dtype=dtype),
        torch.tensor([theta_range[1], thetadot_range[1]], dtype=dtype), 1000)
    x_samples = torch.zeros((theta_thetadot_samples.shape[0], 6), dtype=dtype)
    x_samples[:, [2, 5]] = theta_thetadot_samples
    u_samples = utils.uniform_sample_in_box(
        torch.tensor([u0_range[0], u1_range[0]], dtype=dtype),
        torch.tensor([u0_range[1], u1_range[1]], dtype=dtype), 1000)
    xu_tensors = []
    x_next_tensors = []
    for i in range(x_samples.shape[0]):
        for j in range(u_samples.shape[0]):
            result = scipy.integrate.solve_ivp(
                lambda t, x: plant.dynamics(x, u_samples[j, :].detach().numpy(
                )), (0, dt), x_samples[i, :].detach().numpy())
            xu_tensors.append(
                torch.cat((x_samples[i], u_samples[j])).reshape((1, -1)))
            x_next_tensors.append(
                torch.from_numpy(result.y[:, -1]).reshape((1, -1)))
    dataset_input = torch.cat(xu_tensors, dim=0)
    dataset_output = torch.cat(x_next_tensors, dim=0)
    return torch.utils.data.TensorDataset(dataset_input, dataset_output)


def train_forward_model(forward_model, model_dataset, num_epochs):
    plant = rocket.Rocket()
    dtype = torch.float64
    u_equilibrium = torch.tensor([0, plant.hover_thrust], dtype=dtype)

    xu_inputs, x_next_outputs = model_dataset[:]
    network_input_data = xu_inputs[:, [2, 5, 6, 7]]
    network_output_data = x_next_outputs[:, 3:] - xu_inputs[:, 3:6]
    v_dataset = torch.utils.data.TensorDataset(network_input_data,
                                               network_output_data)

    def compute_next_v(model, theta_thetadot_u):
        return model(theta_thetadot_u) - model(
            torch.cat((torch.tensor([0, 0], dtype=dtype), u_equilibrium)))

    utils.train_approximator(v_dataset,
                             forward_model,
                             compute_next_v,
                             batch_size=50,
                             num_epochs=num_epochs,
                             lr=0.001)


def train_lqr_value_approximator(lyapunov_relu, V_lambda, R, x_equilibrium,
                                 x_lo, x_up, num_samples, lqr_S: torch.Tensor,
                                 num_epochs):
    """
    We train both lyapunov_relu and R such that ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁
    approximates the lqr cost-to-go.
    """
    x_samples = utils.uniform_sample_in_box(x_lo, x_up, num_samples)
    V_samples = torch.sum((x_samples.T - x_equilibrium.reshape(
        (6, 1))) * (lqr_S @ (x_samples.T - x_equilibrium.reshape((6, 1)))),
                          dim=0).reshape((-1, 1))
    state_value_dataset = torch.utils.data.TensorDataset(x_samples, V_samples)
    R.requires_grad_(True)

    def compute_v(model, x):
        return model(x) - model(x_equilibrium) + V_lambda * torch.norm(
            R @ (x - x_equilibrium.reshape((1, 6))).T, p=1, dim=0).reshape(
                (-1, 1))

    utils.train_approximator(state_value_dataset,
                             lyapunov_relu,
                             compute_v,
                             batch_size=50,
                             num_epochs=num_epochs,
                             lr=0.001,
                             additional_variable=[R])
    R.requires_grad_(False)


def train_lqr_control_approximator(controller_relu, x_equilibrium,
                                   u_equilibrium, x_lo, x_up, num_samples,
                                   lqr_K: torch.Tensor, num_epochs):
    x_samples = utils.uniform_sample_in_box(x_lo, x_up, num_samples)
    u_samples = (lqr_K @ (x_samples.T - x_equilibrium.reshape(
        (6, 1))) + u_equilibrium.reshape((2, 1))).T
    state_control_dataset = torch.utils.data.TensorDataset(
        x_samples, u_samples)

    def compute_u(model, x):
        return model(x) - model(x_equilibrium) + u_equilibrium

    utils.train_approximator(state_control_dataset,
                             controller_relu,
                             compute_u,
                             batch_size=50,
                             num_epochs=num_epochs,
                             lr=0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rocket training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument("--load_dynamics_data", type=str, default=None)
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--load_forward_model", type=str, default=None)
    parser.add_argument("--load_lyapunov_relu", type=str, default=None)
    parser.add_argument("--load_controller_relu", type=str, default=None)
    parser.add_argument("--train_lqr_approximator", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--search_R", action="store_true")
    parser.add_argument("--train_on_samples", action="store_true")
    parser.add_argument("--train_adversarial", action="store_true")
    args = parser.parse_args()
    dt = 0.01
    dtype = torch.float64

    if args.generate_dynamics_data:
        dynamics_dataset = generate_dynamics_data(dt)

    if args.load_dynamics_data:
        dynamics_dataset = torch.load(args.load_dynamics_data)

    if args.train_forward_model:
        forward_relu = utils.setup_relu((4, 6, 6, 3),
                                        params=None,
                                        bias=True,
                                        negative_slope=0.01,
                                        dtype=dtype)
        train_forward_model(forward_relu, dynamics_dataset, num_epochs=100)

    if args.load_forward_model:
        forward_model_data = torch.load(args.load_forward_model)
        forward_relu = utils.setup_relu(
            forward_model_data["linear_layer_width"],
            params=None,
            negative_slope=forward_model_data["negative_slope"],
            bias=forward_model_data["bias"],
            dtype=dtype)
        forward_relu.load_state_dict(forward_model_data["state_dict"])

    plant = rocket.Rocket()
    x_star = np.zeros((6, ))
    u_star = np.array([0, plant.hover_thrust])

    lqr_Q = np.diag([10, 10, 10, 1, 1, plant.length / 2. / np.pi])
    lqr_R = np.diag([0.2, 0.1])
    plant_A, plant_B = plant.linearized_dynamics(x_star, u_star)
    lqr_S = scipy.linalg.solve_continuous_are(plant_A, plant_B, lqr_Q, lqr_R)
    lqr_K = -np.linalg.solve(lqr_R, plant_B.T @ lqr_S)
    R = torch.cat((torch.tensor(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
        dtype=dtype), torch.from_numpy(lqr_S)),
                  dim=0)

    lyapunov_relu = utils.setup_relu((6, 10, 8, 6, 1),
                                     params=None,
                                     negative_slope=0.1,
                                     bias=True,
                                     dtype=dtype)
    V_lambda = 0.9
    if args.load_lyapunov_relu:
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

    controller_relu = utils.setup_relu((6, 6, 6, 2),
                                       params=None,
                                       negative_slope=0.01,
                                       bias=True,
                                       dtype=dtype)
    if args.load_controller_relu:
        controller_data = torch.load(args.load_controller_relu)
        controller_relu = utils.setup_relu(
            controller_data["linear_layer_width"],
            params=None,
            negative_slope=controller_data["negative_slope"],
            bias=controller_data["bias"],
            dtype=dtype)
        controller_relu.load_state_dict(controller_data["state_dict"])

    q_equilibrium = torch.tensor([0, 0, 0], dtype=dtype)
    u_equilibrium = torch.tensor([0, plant.hover_thrust], dtype=dtype)

    x_lo = torch.tensor([-0.05, -0.05, -np.pi * 0.05, -0.2, -0.2, -0.1],
                        dtype=dtype) / 2
    x_up = torch.tensor([0.05, 0.05, np.pi * 0.05, 0.2, 0.2, 0.1],
                        dtype=dtype) * 2
    u_lo = torch.tensor([-0.15 * plant.hover_thrust, 0], dtype=dtype)
    u_up = torch.tensor([0.15 * plant.hover_thrust, 3 * plant.hover_thrust],
                        dtype=dtype)

    if args.train_lqr_approximator:
        x_equilibrium = torch.cat(
            (q_equilibrium, torch.zeros((3, ), dtype=dtype)))
        train_lqr_control_approximator(controller_relu,
                                       x_equilibrium,
                                       u_equilibrium,
                                       x_lo,
                                       x_up,
                                       100000,
                                       torch.from_numpy(lqr_K),
                                       num_epochs=20)
        train_lqr_value_approximator(lyapunov_relu,
                                     V_lambda,
                                     R,
                                     x_equilibrium,
                                     x_lo,
                                     x_up,
                                     100000,
                                     torch.from_numpy(lqr_S),
                                     num_epochs=50)

    forward_system = relu_system.ReLUSecondOrderResidueSystemGivenEquilibrium(
        dtype,
        x_lo,
        x_up,
        u_lo,
        u_up,
        forward_relu,
        q_equilibrium,
        u_equilibrium,
        dt,
        network_input_x_indices=[2, 5])
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy())
    lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(closed_loop_system,
                                                     lyapunov_relu)

    if args.search_R:
        _, R_sigma, _ = np.linalg.svd(R.detach().numpy())
        R_options = r_options.SearchRwithSVDOptions(R.shape, R_sigma * 0.8)
        R_options.set_variable_value(R.detach().numpy())
    else:
        R_options = r_options.FixedROptions(R)
    dut = train_lyapunov.TrainLyapunovReLU(lyap, V_lambda,
                                           closed_loop_system.x_equilibrium,
                                           R_options)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-5
    dut.lyapunov_positivity_convergence_tol = 5e-6
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.1
    dut.lyapunov_derivative_epsilon = 0.001
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    state_samples_all = utils.get_meshgrid_samples(x_lo,
                                                   x_up, (7, 7, 7, 7, 7, 7),
                                                   dtype=dtype)
    dut.output_flag = True
    if args.train_on_samples:
        dut.train_lyapunov_on_samples(state_samples_all,
                                      num_epochs=10,
                                      batch_size=50)
    dut.enable_wandb = args.enable_wandb
    if args.train_adversarial:
        options = train_lyapunov.TrainLyapunovReLU.AdversarialTrainingOptions()
        options.num_batches = 10
        options.num_epochs_per_mip = 10
        options.positivity_samples_pool_size = 10000
        options.derivative_samples_pool_size = 30000
        dut.lyapunov_positivity_mip_pool_solutions = 100
        dut.lyapunov_derivative_mip_pool_solutions = 500
        dut.add_derivative_adversarial_state = True
        dut.add_positivity_adversarial_state = True
        positivity_state_samples_init = utils.uniform_sample_in_box(
            x_lo, x_up, 1000)
        derivative_state_samples_init = positivity_state_samples_init
        result = dut.train_adversarial(positivity_state_samples_init,
                                       derivative_state_samples_init, options)
    else:
        dut.train(torch.empty((0, 6), dtype=dtype))
    pass
