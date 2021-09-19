import neural_network_lyapunov.examples.pendulum.pendulum as pendulum
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.control_lyapunov as control_lyapunov
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.r_options as r_options

import torch
import numpy as np
import argparse


def generate_pendulum_dynamics_data():
    dtype = torch.float64
    plant = pendulum.Pendulum(dtype)

    xu_samples = utils.uniform_sample_in_box(
        torch.tensor([-0.5 * np.pi, -5, -10], dtype=dtype),
        torch.tensor([2.5 * np.pi, 5, 10], dtype=dtype), 100000)
    xdot = torch.vstack([
        plant.dynamics(xu_samples[i, :2], xu_samples[i, 2:])
        for i in range(xu_samples.shape[0])
    ])
    return torch.utils.data.TensorDataset(xu_samples, xdot)


def train_forward_model(dynamics_model: control_affine_system.
                        ReluSecondOrderControlAffineSystem, model_dataset):
    (xu_inputs, x_next_outputs) = model_dataset[:]
    v_dataset = torch.utils.data.TensorDataset(
        xu_inputs, x_next_outputs[:, 1].reshape((-1, 1)))

    def compute_vdot(phi_a, state_action, phi_b):
        x, u = torch.split(state_action, [2, 1], dim=1)
        vdot = phi_a(x) + (phi_b(x).reshape((-1, 1)) * u).reshape((-1, 1))
        return vdot

    utils.train_approximator(v_dataset,
                             dynamics_model.phi_a,
                             compute_vdot,
                             batch_size=30,
                             num_epochs=100,
                             lr=0.001,
                             additional_variable=list(
                                 dynamics_model.phi_b.parameters()),
                             output_fun_args=dict(phi_b=dynamics_model.phi_b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pendulum clf training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
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
                        help="path to load the forward model")
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--enable_wandb", action="store_true")
    args = parser.parse_args()
    dtype = torch.float64

    if args.generate_dynamics_data:
        model_dataset = generate_pendulum_dynamics_data()
    if args.load_dynamics_data is not None:
        model_dataset = torch.load(args.load_dynamics_data)

    phi_a = utils.setup_relu((2, 6, 4, 1),
                             params=None,
                             negative_slope=0.01,
                             bias=True,
                             dtype=dtype)
    phi_b = utils.setup_relu((2, 6, 4, 1),
                             params=None,
                             negative_slope=0.01,
                             bias=True,
                             dtype=dtype)
    x_lo = torch.tensor([np.pi - 0.5 * np.pi, -3], dtype=dtype)
    x_up = torch.tensor([np.pi + 0.5 * np.pi, 3], dtype=dtype)
    u_lo = torch.tensor([-10], dtype=dtype)
    u_up = torch.tensor([10], dtype=dtype)
    dynamics_model = control_affine_system.ReluSecondOrderControlAffineSystem(
        x_lo,
        x_up,
        u_lo,
        u_up,
        phi_a,
        phi_b,
        method=mip_utils.PropagateBoundsMethod.IA)

    if args.train_forward_model is not None:
        train_forward_model(dynamics_model, model_dataset)
        linear_layer_width_a, negative_slope_a, bias_a = \
            utils.extract_relu_structure(phi_a)
        linear_layer_width_b, negative_slope_b, bias_b = \
            utils.extract_relu_structure(phi_b)
        torch.save(
            {
                "phi_a": {
                    "linear_layer_width": linear_layer_width_a,
                    "state_dict": phi_a.state_dict(),
                    "negative_slope": negative_slope_a,
                    "bias": bias_a
                },
                "phi_b": {
                    "linear_layer_width": linear_layer_width_b,
                    "state_dict": phi_b.state_dict(),
                    "negative_slope": negative_slope_b,
                    "bias": bias_b
                }
            }, args.train_forward_model)

    elif args.load_forward_model is not None:
        dynamics_model_data = torch.load(args.load_forward_model)
        phi_a = utils.setup_relu(
            dynamics_model_data["phi_a"]["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["phi_a"]["negative_slope"],
            bias=dynamics_model_data["phi_a"]["bias"],
            dtype=dtype)
        phi_a.load_state_dict(dynamics_model_data["phi_a"]["state_dict"])
        phi_b = utils.setup_relu(
            dynamics_model_data["phi_b"]["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["phi_b"]["negative_slope"],
            bias=dynamics_model_data["phi_b"]["bias"],
            dtype=dtype)
        phi_b.load_state_dict(dynamics_model_data["phi_b"]["state_dict"])

    V_lambda = 0.5
    lyapunov_relu = utils.setup_relu((2, 8, 8, 1),
                                     params=None,
                                     negative_slope=0.1,
                                     bias=True,
                                     dtype=dtype)
    R = torch.tensor([[0.5, 1], [-1, 0], [1, 1]], dtype=dtype)

    lyapunov_hybrid_system = control_lyapunov.ControlLyapunov(
        dynamics_model, lyapunov_relu)

    R_options = r_options.SearchRwithSVDOptions(R.shape, np.array([0.1, 0.2]))
    R_options.set_variable_value(R.detach().numpy())

    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, None, x_lo, x_up,
                                        u_lo, u_up)

    x_equilibrium = torch.tensor([np.pi, 0], dtype=dtype)
    dut = train_lyapunov.TrainLyapunovReLU(lyapunov_hybrid_system, V_lambda,
                                           x_equilibrium, R_options)

    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_clamp_min = 0.
    dut.output_flag = True
    dut.max_iterations = args.max_iterations
    dut.enable_wandb = args.enable_wandb
    state_samples_all = torch.empty((0, 2), dtype=dtype)
    dut.train(state_samples_all)

    pass
