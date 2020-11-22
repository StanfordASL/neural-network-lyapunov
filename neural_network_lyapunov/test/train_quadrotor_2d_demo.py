import neural_network_lyapunov.test.quadrotor_2d as quadrotor_2d
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.train_lyapunov as train_lyapunov
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
    u_equilibrium = plant.u_equilibrium

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
    parser.add_argument("--load_forward_model", type=str, default=None,
                        help="path to load dynamics model")
    parser.add_argument("--load_lyapunov_relu", type=str, default=None,
                        help="path to the lyapunov model data.")
    parser.add_argument("--load_controller_relu", type=str, default=None,
                        help="path to the controller data.")
    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dt = 0.01
    dtype = torch.float64
    if args.generate_dynamics_data:
        model_dataset = generate_quadrotor_dynamics_data(dt)

    if args.load_dynamics_data is not None:
        model_dataset = torch.load(args.load_dynamics_data)

    forward_model = utils.setup_relu(
        (3, 5, 5, 3), params=None, bias=True, negative_slope=0.01,
        dtype=dtype)
    if args.train_forward_model:
        train_forward_model(forward_model, model_dataset)

    if args.load_forward_model:
        forward_model_data = torch.load(args.load_forward_model)
        forward_model = utils.setup_relu(
            forward_model_data["linear_layer_width"], params=None,
            bias=forward_model_data["bias"],
            negative_slope=forward_model_data["negative_slope"], dtype=dtype)
        forward_model.load_state_dict(forward_model_data["state_dict"])

    plant = quadrotor_2d.Quadrotor2D(dtype)
    x_star = np.zeros((6,))
    u_star = plant.u_equilibrium.detach().numpy()
    lqr_Q = np.diag([10, 10, 10, 1, 1, plant.length/2./np.pi])
    lqr_R = np.array([[0.1, 0.05], [0.05, 0.1]])
    K, S = plant.lqr_control(lqr_Q, lqr_R, x_star, u_star)
    S_eig_value, S_eig_vec = np.linalg.eig(S)

    # R = torch.zeros((9, 6), dtype=dtype)
    # R[:3, :3] = torch.eye(3, dtype=dtype)
    # R[3:6, :3] = torch.eye(3, dtype=dtype) / np.sqrt(2)
    # R[3:6, 3:6] = torch.eye(3, dtype=dtype) / np.sqrt(2)
    # R[6:9, :3] = -torch.eye(3, dtype=dtype) / np.sqrt(2)
    # R[6:9, 3:6] = torch.eye(3, dtype=dtype) / np.sqrt(2)
    # R = torch.cat((R, torch.from_numpy(S_eig_vec)), dim=0)
    # R = torch.from_numpy(S_eig_vec)
    R = torch.eye(6, dtype=dtype)

    lyapunov_relu = utils.setup_relu(
        (6, 8, 8, 1), params=None, negative_slope=0.1, bias=True, dtype=dtype)
    V_lambda = 0.9
    if args.load_lyapunov_relu is not None:
        lyapunov_data = torch.load(args.load_lyapunov_relu)
        lyapunov_relu = utils.setup_relu(
            lyapunov_data["linear_layer_width"], params=None,
            negative_slope=lyapunov_data["negative_slope"],
            bias=lyapunov_data["bias"], dtype=dtype)
        lyapunov_relu.load_state_dict(lyapunov_data["state_dict"])
        V_lambda = lyapunov_data["V_lambda"]
        R = lyapunov_data["R"]

    controller_relu = utils.setup_relu(
        (6, 6, 4, 2), params=None, negative_slope=0.01, bias=True, dtype=dtype)
    if args.load_controller_relu is not None:
        controller_data = torch.load(args.load_controller_relu)
        controller_relu = utils.setup_relu(
            controller_data["linear_layer_width"], params=None,
            negative_slope=controller_data["negative_slope"],
            bias=controller_data["bias"], dtype=dtype)
        controller_relu.load_state_dict(controller_data["state_dict"])

    q_equilibrium = torch.tensor([0, 0, 0], dtype=dtype)
    u_equilibrium = plant.u_equilibrium
    x_lo = torch.tensor(
        [-0.1, -0.1, -np.pi * 0.1, -0.5, -0.5, -0.3], dtype=dtype)
    x_up = -x_lo
    u_lo = torch.tensor([-15, -15], dtype=dtype)
    u_up = torch.tensor([25, 25], dtype=dtype)
    forward_system = relu_system.ReLUSecondOrderResidueSystemGivenEquilibrium(
        dtype, x_lo, x_up, u_lo, u_up, forward_model, q_equilibrium,
        u_equilibrium, dt, network_input_x_indices=[2])
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium, u_lo.detach().numpy(),
        u_up.detach().numpy())
    lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(
        closed_loop_system, lyapunov_relu)

    dut = train_lyapunov.TrainLyapunovReLU(
        lyap, V_lambda, closed_loop_system.x_equilibrium, R)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-5
    dut.max_iterations = 2000
    dut.lyapunov_positivity_epsilon = 0.5
    dut.lyapunov_derivative_epsilon = 0.001
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    state_samples_all = utils.get_meshgrid_samples(
        x_lo, x_up, (7, 7, 7, 7, 7, 7), dtype=dtype)
    dut.output_flag = True
    dut.train_lyapunov_on_samples(
        state_samples_all, num_epochs=10, batch_size=50)
    dut.enable_wandb = True
    dut.train(torch.empty((0, 6), dtype=dtype))
    pass
