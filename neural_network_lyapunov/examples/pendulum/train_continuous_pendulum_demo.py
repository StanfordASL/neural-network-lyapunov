import neural_network_lyapunov.examples.pendulum.pendulum as pendulum
import neural_network_lyapunov.train_lyapunov_barrier as train_lyapunov_barrier
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.continuous_time_lyapunov as \
    continuous_time_lyapunov
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.r_options as r_options

import torch
import numpy as np
import argparse
import scipy.integrate


def generate_dynamics_data():
    dtype = torch.float64
    plant = pendulum.Pendulum(dtype)
    x_lo = torch.tensor([-0.5 * np.pi, -5.], dtype=dtype)
    x_up = torch.tensor([2.5 * np.pi, 5.], dtype=dtype)
    u_lo = torch.tensor([-20], dtype=dtype)
    u_up = torch.tensor([20], dtype=dtype)
    xu_samples = utils.uniform_sample_in_box(torch.cat((x_lo, u_lo)),
                                             torch.cat((x_up, u_up)), 100000)
    xdot = torch.empty((xu_samples.shape[0], 2), dtype=dtype)
    with torch.no_grad():
        for i in range(xu_samples.shape[0]):
            xdot[i] = plant.dynamics(xu_samples[i, :2], xu_samples[i, 2:])
    return torch.utils.data.TensorDataset(xu_samples, xdot)


def train_forward_model(dynamics_relu, model_dataset):
    xu_inputs, xdot_outputs = model_dataset[:]
    thetaddot_dataset = torch.utils.data.TensorDataset(
        xu_inputs, xdot_outputs[:, 1].reshape((-1, 1)))

    def compute_thetaddot(model, state_action):
        return model(state_action) - model(
            torch.tensor([np.pi, 0, 0], dtype=torch.float64))

    utils.train_approximator(thetaddot_dataset,
                             dynamics_relu,
                             compute_thetaddot,
                             batch_size=50,
                             num_epochs=200,
                             lr=0.005)


def simulate_system(plant, controller_relu, u_lo, u_up, lyapunov_hybrid_system,
                    V_lambda, R, initial_state, T):
    def closed_loop_dynamics(t, x):
        with torch.no_grad():
            x_torch = torch.from_numpy(x)
            u = torch.clamp(
                controller_relu(x_torch) -
                controller_relu(torch.tensor([np.pi, 0], dtype=torch.float64)),
                u_lo, u_up).detach().numpy()
            xdot = plant.dynamics(x, u)
            return xdot

    result = scipy.integrate.solve_ivp(closed_loop_dynamics, [0, T],
                                       initial_state)
    with torch.no_grad():
        V_val = lyapunov_hybrid_system.lyapunov_value(
            torch.from_numpy(result.y.T),
            torch.tensor([np.pi, 0], dtype=torch.float64),
            V_lambda,
            R=R)
    return result, V_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pendulum training demo")
    parser.add_argument("--generate_dynamics_data",
                        type=str,
                        default=None,
                        help="path to save the generated dynamics data")
    parser.add_argument("--load_dynamics_data",
                        type=str,
                        default=None,
                        help="path to load dynamics data")
    parser.add_argument("--train_forward_model",
                        type=str,
                        default=None,
                        help="path to save trained forward model")
    parser.add_argument("--load_forward_model",
                        type=str,
                        default=None,
                        help="path to load trained forward model")
    parser.add_argument("--load_lyapunov_relu",
                        type=str,
                        default=None,
                        help="path to load lyapunov relu")
    parser.add_argument("--load_controller_relu",
                        type=str,
                        default=None,
                        help="path to load controller relu")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--max_iterations",
                        type=int,
                        default=500,
                        help="number of iterations in training Lyapunov")
    args = parser.parse_args()

    dtype = torch.float64

    if args.generate_dynamics_data:
        model_dataset = generate_dynamics_data()
        torch.save(model_dataset, args.generate_dynamics_data)
    if args.load_dynamics_data:
        model_dataset = torch.load(args.load_dynamics_data)

    dynamics_relu = utils.setup_relu((3, 5, 5, 1),
                                     params=None,
                                     negative_slope=0.01,
                                     bias=True,
                                     dtype=dtype)
    if args.train_forward_model is not None:
        train_forward_model(dynamics_relu, model_dataset)
        torch.save({
            "state_dict": dynamics_relu.state_dict(),
        }, args.train_forward_model)

    if args.load_forward_model is not None:
        dynamics_model_data = torch.load(args.load_forward_model)
        dynamics_relu.load_state_dict(dynamics_model_data["state_dict"])

    x_lo = torch.tensor([-0.1 * np.pi, -5], dtype=dtype)
    x_up = torch.tensor([2.1 * np.pi, 5], dtype=dtype)
    u_lo = torch.tensor([-20], dtype=dtype)
    u_up = torch.tensor([20], dtype=dtype)
    dynamics_model = pendulum.PendulumReluContinuousTime(
        dtype, x_lo, x_up, u_lo, u_up, dynamics_relu)

    if args.load_controller_relu:
        controller_data = torch.load(args.load_controller_relu)
        controller_relu = utils.setup_relu(
            controller_data["linear_layer_width"],
            params=None,
            negative_slope=controller_data["negative_slope"],
            bias=controller_data["bias"],
            dtype=dtype)
        controller_relu.load_state_dict(controller_data["state_dict"])
    else:
        controller_relu = utils.setup_relu((2, 3, 2, 1),
                                           params=None,
                                           negative_slope=0.1,
                                           bias=True,
                                           dtype=dtype)
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
    else:
        lyapunov_relu = utils.setup_relu((2, 8, 8, 6, 1),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=dtype)
        V_lambda = 0.5
        R = torch.tensor([[1, -1], [1, 1], [0.5, 0.3]], dtype=dtype)
    closed_loop_system = feedback_system.FeedbackSystem(
        dynamics_model, controller_relu, dynamics_model.x_equilibrium,
        dynamics_model.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy())
    lyapunov_hybrid_system = \
        continuous_time_lyapunov.LyapunovContinuousTimeSystem(
            closed_loop_system, lyapunov_relu)
    R_options = r_options.FixedROptions(R)

    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, controller_relu,
                                        x_lo, x_up, u_lo, u_up)

    dut = train_lyapunov_barrier.Trainer()
    dut.add_lyapunov(lyapunov_hybrid_system, V_lambda,
                     closed_loop_system.x_equilibrium, R_options)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-5
    dut.lyapunov_positivity_convergence_tol = 1E-5
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.2
    dut.lyapunov_derivative_epsilon = 0.5
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    state_samples_all = utils.get_meshgrid_samples(x_lo,
                                                   x_up, (51, 51),
                                                   dtype=torch.float64)
    dut.output_flag = True
    dut.enable_wandb = args.enable_wandb
    dut.train(torch.empty((0, 2), dtype=dtype))

    pass
