import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.examples.quadrotor3d.quadrotor as quadrotor
import neural_network_lyapunov.examples.quadrotor3d.control_affine_quadrotor \
    as control_affine_quadrotor
import neural_network_lyapunov.utils as utils

import torch
import numpy as np
import argparse


def generate_quadrotor_dynamics_data():
    dtype = torch.float64
    plant = quadrotor.Quadrotor(dtype)
    x_lo = torch.tensor([
        0, 0, 0, -np.pi, -0.4 * np.pi, -np.pi, 0, 0, 0, -np.pi, -np.pi, -np.pi
    ],
                        dtype=dtype)
    x_up = -x_lo
    u_lo = torch.zeros((4, ), dtype=dtype)
    u_up = plant.hover_thrust * 3 * torch.ones((4, ), dtype=dtype)
    xu_samples = utils.uniform_sample_in_box(torch.cat((x_lo, u_lo)),
                                             torch.cat((x_up, u_up)), 1000000)
    xdot = torch.vstack([
        plant.dynamics(xu_samples[i, :12], xu_samples[i, 12:])
        for i in range(xu_samples.shape[0])
    ])
    return torch.utils.data.TensorDataset(xu_samples, xdot)


def train_phi_a(phi_a, model_dataset):
    # Train phi_a(x) network in the forward dynamics.
    (xu_inputs, xdot_outputs) = model_dataset[:]

    def compute_rpydot(phi_a, state_action):
        x, u = torch.split(state_action, [12, 4], dim=1)
        rpy = x[:, 3:6]
        omega = x[:, 9:12]
        rpy_dot = phi_a(torch.cat((rpy, omega), dim=1)) - phi_a(
            torch.cat(
                (rpy, torch.zeros_like(omega, dtype=omega.dtype)), dim=1))
        return rpy_dot

    rpydot_dataset = torch.utils.data.TensorDataset(xu_inputs,
                                                    xdot_outputs[:, 3:6])
    utils.train_approximator(rpydot_dataset,
                             phi_a,
                             compute_rpydot,
                             batch_size=50,
                             num_epochs=200,
                             lr=0.001)


def train_phi_b(phi_b, model_dataset, u_equilibrium):
    # Train phi_b(x) network in the forward dynamics.
    (xu_inputs, xdot_outputs) = model_dataset[:]

    def compute_pos_ddot(phi_b, state_action):
        x, u = torch.split(state_action, [12, 4], dim=1)
        rpy = x[:, 3:6]
        pos_ddot = phi_b(rpy) * (torch.sum(u, dim=1).repeat(3, 1).T) - phi_b(
            torch.zeros((3, ), dtype=rpy.dtype)) * torch.sum(u_equilibrium)
        return pos_ddot

    pos_ddot_dataset = torch.utils.data.TensorDataset(xu_inputs,
                                                      xdot_outputs[:, 6:9])
    utils.train_approximator(pos_ddot_dataset,
                             phi_b,
                             compute_pos_ddot,
                             batch_size=50,
                             num_epochs=200,
                             lr=0.001)


def train_phi_c(phi_c, C, model_dataset, u_equilibrium):
    # Train phi_c(x) network and C in the forward dynamics.
    (xu_inputs, xdot_outputs) = model_dataset[:]

    def compute_omega_dot(phi_c, state_action, C):
        x, u = torch.split(state_action, [12, 4], dim=1)
        omega = x[:, 9:12]
        omega_dot = phi_c(omega) - phi_c(torch.zeros(
            (3, ), dtype=omega.dtype)) - C @ u_equilibrium + (C @ u.T).T
        return omega_dot

    omega_dot_dataset = torch.utils.data.TensorDataset(xu_inputs,
                                                       xdot_outputs[:, 9:12])
    C.requires_grad = True
    utils.train_approximator(omega_dot_dataset,
                             phi_c,
                             compute_omega_dot,
                             batch_size=50,
                             num_epochs=200,
                             lr=0.001,
                             additional_variable=[C],
                             output_fun_args={"C": C})

    C.requires_grad = False


def train_forward_model(
        dynamics_model: control_affine_quadrotor.ControlAffineQuadrotor,
        model_dataset):
    train_phi_a(dynamics_model.phi_a, model_dataset)
    train_phi_b(dynamics_model.phi_b, model_dataset,
                dynamics_model.u_equilibrium)
    train_phi_c(dynamics_model.phi_c, dynamics_model.C, model_dataset,
                dynamics_model.u_equilibrium)


def save_dynamics_model(phi_a, phi_b, phi_c, C, savepath):
    linear_layer_width_a, negative_slope_a, bias_a = \
        utils.extract_relu_structure(phi_a)
    linear_layer_width_b, negative_slope_b, bias_b = \
        utils.extract_relu_structure(phi_b)
    linear_layer_width_c, negative_slope_c, bias_c = \
        utils.extract_relu_structure(phi_c)
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
            },
            "phi_c": {
                "linear_layer_width": linear_layer_width_c,
                "state_dict": phi_c.state_dict(),
                "negative_slope": negative_slope_c,
                "bias": bias_c
            },
            "C": C
        }, savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="quadrotor clf training demo")
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
    parser.add_argument("--load_lyapunov_relu",
                        type=str,
                        default=None,
                        help="path to load lyapunov")
    args = parser.parse_args()

    dtype = torch.float64
    plant = quadrotor.Quadrotor(dtype)
    if args.generate_dynamics_data:
        model_dataset = generate_quadrotor_dynamics_data()
    if args.load_dynamics_data is not None:
        model_dataset = torch.load(args.load_dynamics_data)

    x_lo = torch.tensor([
        -0.5, -0.5, -0.5, -np.pi, -0.4 * np.pi, -np.pi, -1, -1, -1, -np.pi,
        -np.pi, -np.pi
    ],
                        dtype=dtype)
    x_up = -x_lo
    u_lo = torch.zeros((4, ), dtype=dtype)
    u_up = torch.ones((4, ), dtype=dtype) * 3 * plant.hover_thrust
    u_equilibrium = torch.ones((4, ), dtype=dtype) * plant.hover_thrust
    if args.train_forward_model is not None:
        phi_a = utils.setup_relu((6, 12, 9, 3),
                                 params=None,
                                 negative_slope=0.01,
                                 bias=True,
                                 dtype=dtype)
        phi_b = utils.setup_relu((3, 10, 10, 3),
                                 params=None,
                                 negative_slope=0.01,
                                 bias=True,
                                 dtype=dtype)
        phi_c = utils.setup_relu((3, 6, 6, 3),
                                 params=None,
                                 negative_slope=0.1,
                                 bias=True,
                                 dtype=dtype)
        C = torch.tensor(
            [[0.5, -0.5, 0.5, -0.5], [1, 2, -1, 2], [1, 0.5, 0.5, -1]],
            dtype=dtype)
        dynamics_model = control_affine_quadrotor.ControlAffineQuadrotor(
            x_lo, x_up, u_lo, u_up, phi_a, phi_b, phi_c, C, u_equilibrium,
            mip_utils.PropagateBoundsMethod.IA)
        train_forward_model(dynamics_model, model_dataset)
        save_dynamics_model(phi_a, phi_b, phi_c, C, args.train_forward_model)
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
        phi_c = utils.setup_relu(
            dynamics_model_data["phi_c"]["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["phi_c"]["negative_slope"],
            bias=dynamics_model_data["phi_c"]["bias"],
            dtype=dtype)
        phi_c.load_state_dict(dynamics_model_data["phi_c"]["state_dict"])
        C = dynamics_model_data["C"]
        dynamics_model = control_affine_quadrotor.ControlAffineQuadrotor(
            x_lo, x_up, u_lo, u_up, phi_a, phi_b, phi_c, C, u_equilibrium,
            mip_utils.PropagateBoundsMethod.IA)
    pass
