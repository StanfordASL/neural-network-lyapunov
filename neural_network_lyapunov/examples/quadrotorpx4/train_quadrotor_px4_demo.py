import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.examples.quadrotor3d.quadrotor as quadrotor

import torch
import numpy as np
import scipy.integrate
import argparse
import os


def train_forward_model(forward_model, xu_equilibrium, model_dataset,
                        num_epochs):
    # The forward model maps (dx[n], dy[n], dz[n], roll[n], pitch[n], yaw[n],
    # roll_sp[n], pitch_sp[n], yaw_sp[n], thrust_sp[n]) to
    # (dx[n+1] - dx[n], dy[n+1] - dy[n], dz[n+1] - dz[n], roll[n+1] - roll[n],
    # pitch[n+1] - pitch[n], yaw[n+1] - yaw[n])
    network_input_data, network_output_data = model_dataset[:]
    v_dataset = torch.utils.data.TensorDataset(
        network_input_data, network_output_data)

    def compute_next_v(model, xu):
        return model(xu) - model(xu_equilibrium)
    utils.train_approximator(
        v_dataset, forward_model, compute_next_v, batch_size=50,
        num_epochs=num_epochs, lr=0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PX4 quadrotor training demo")
    parser.add_argument("load_dynamics_data", type=str,
                        help="path to the dynamics data.")
    parser.add_argument("--load_forward_model", type=str, default=None,
                        help="path to load dynamics model")
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--save_forward_model", type=str, default=None,
                        help="path to save dynamics model")
    parser.add_argument("--load_lyapunov", type=str, default=None,
                        help="path to load lyapunov model")
    parser.add_argument("--save_lyapunov", type=str, default=None,
                        help="path to save lyapunov model")
    parser.add_argument("--load_controller", type=str, default=None,
                        help="path to load controller model")
    parser.add_argument("--save_controller", type=str, default=None,
                        help="path to save controller model")
    parser.add_argument("--train_on_samples", action="store_true")
    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dtype = torch.float64

    model_dataset = torch.load(args.load_dynamics_data)

    dt = 1./30.
    hover_thrust = .705
    xu_equilibrium = torch.tensor(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., hover_thrust], dtype=dtype)
    R = 0.1 * torch.eye(9, dtype=dtype)
    V_lambda = 0.9
    x_lo = torch.tensor(
        [-5, -5, 0, -np.pi, -np.pi, -np.pi, -5, -5, -5], dtype=dtype)
    x_up = torch.tensor(
        [5, 5, 5, np.pi, np.pi, np.pi, 5, 5, 5], dtype=dtype)
    u_lo = torch.tensor([-np.pi, -np.pi, -np.pi, 0.], dtype=dtype)
    u_up = torch.tensor([np.pi, np.pi, np.pi, 1.], dtype=dtype)

    if args.load_forward_model:
        forward_model = torch.load(args.load_forward_model)
    else:
        forward_model = utils.setup_relu(
            (10, 15, 15, 6), params=None, bias=True, negative_slope=0.01,
            dtype=dtype)
    if args.train_forward_model:
        train_forward_model(forward_model, xu_equilibrium, model_dataset,
                            num_epochs=100)
    if args.save_forward_model:
        torch.save(forward_model, args.save_forward_model)

    if args.load_lyapunov:
        lyapunov_relu = torch.load(args.load_lyapunov)
    else:
        lyapunov_relu = utils.setup_relu(
            (9, 6, 1), params=None, negative_slope=0.1, bias=True,
            dtype=dtype)

    if args.load_controller:
        controller_relu = torch.load(args.load_controller)
    else:
        controller_relu = utils.setup_relu(
            (9, 6, 4), params=None, negative_slope=0.01, bias=True,
            dtype=dtype)

    forward_system = quadrotor.QuadrotorWithPixhawkReLUSystem(
        dtype, x_lo, x_up, u_lo, u_up, forward_model, hover_thrust, dt)
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium, u_lo.detach().numpy(),
        u_up.detach().numpy())
    lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(
        closed_loop_system, lyapunov_relu)
    R_options = train_lyapunov.SearchROptions(R.shape, 0.01)
    R_options.set_variable_value(R.detach().numpy())
    dut = train_lyapunov.TrainLyapunovReLU(
        lyap, V_lambda, closed_loop_system.x_equilibrium, R_options)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-5
    dut.lyapunov_positivity_convergence_tol = 5e-6
    dut.max_iterations = 100
    dut.lyapunov_positivity_epsilon = 0.1
    dut.lyapunov_derivative_epsilon = 0.003
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    state_samples_all = utils.get_meshgrid_samples(
        x_lo, x_up, (3, 3, 3, 3, 3, 3, 3, 3, 3), dtype=dtype)
    dut.output_flag = True
    if args.train_on_samples:
        dut.train_lyapunov_on_samples(
            state_samples_all, num_epochs=10, batch_size=50)
    dut.enable_wandb = True
    dut.train(torch.empty((0, 9), dtype=dtype))

    if args.save_lyapunov:
        torch.save(lyapunov_relu, args.save_lyapunov)
        if args.save_lyapunov.endswith(".pt"):
            R_prefix = args.save_lyapunov[:-3]
        else:
            R_prefix = args.save_lyapunov
        torch.save(R, R_prefix + 'R.pt')
    if args.save_controller:
        torch.save(controller_relu, args.save_controller)
    pass
