import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.examples.quadrotor3d.quadrotor as quadrotor
import neural_network_lyapunov.examples.quadrotor3d.control_affine_quadrotor \
    as control_affine_quadrotor
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.control_lyapunov as control_lyapunov
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.train_lyapunov as train_lyapunov

import torch
import numpy as np
import argparse
import gurobipy


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


def train_phi_a(phi_a, model_dataset, num_epochs=200, lr=0.001, formulation=2):
    # Train phi_a(x) network in the forward dynamics.
    (xu_inputs, xdot_outputs) = model_dataset[:]

    def compute_rpydot(phi_a, state_action):
        x, u = torch.split(state_action, [12, 4], dim=1)
        rpy = x[:, 3:6]
        omega = x[:, 9:12]
        if formulation == 1:
            rpy_dot = phi_a(torch.cat((rpy, omega), dim=1)) - phi_a(
                torch.cat(
                    (rpy, torch.zeros_like(omega, dtype=omega.dtype)), dim=1))
        elif formulation == 2:
            rpy_dot = phi_a(torch.cat((rpy, omega), dim=1)) - phi_a(
                torch.zeros((6, ), dtype=omega.dtype))
        return rpy_dot

    rpydot_dataset = torch.utils.data.TensorDataset(xu_inputs,
                                                    xdot_outputs[:, 3:6])
    utils.train_approximator(rpydot_dataset,
                             phi_a,
                             compute_rpydot,
                             batch_size=50,
                             num_epochs=num_epochs,
                             lr=lr)


def train_phi_b(phi_b, model_dataset, u_equilibrium, num_epochs=200, lr=0.001):
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
                             num_epochs=num_epochs,
                             lr=lr)


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
    train_phi_a(dynamics_model.phi_a, model_dataset,
                dynamics_model.formulation)
    train_phi_b(dynamics_model.phi_b, model_dataset,
                dynamics_model.u_equilibrium)
    train_phi_c(dynamics_model.phi_c, dynamics_model.C, model_dataset,
                dynamics_model.u_equilibrium)


def save_dynamics_model(phi_a, phi_b, phi_c, C, formulation, savepath):
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
            "C": C,
            "formulation": formulation
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
                        dtype=dtype) * 0.1
    x_up = -x_lo
    u_lo = torch.zeros((4, ), dtype=dtype)
    u_up = torch.ones((4, ), dtype=dtype) * 3 * plant.hover_thrust
    u_equilibrium = torch.ones((4, ), dtype=dtype) * plant.hover_thrust
    if args.train_forward_model is not None:
        phi_a = utils.setup_relu((6, 12, 12, 6, 3),
                                 params=None,
                                 negative_slope=0.01,
                                 bias=True,
                                 dtype=dtype)
        phi_b = utils.setup_relu((3, 12, 12, 6, 3),
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
            x_lo,
            x_up,
            u_lo,
            u_up,
            phi_a,
            phi_b,
            phi_c,
            C,
            u_equilibrium,
            mip_utils.PropagateBoundsMethod.IA,
            formulation=2)
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
        formulation = dynamics_model_data["formulation"]
        dynamics_model = control_affine_quadrotor.ControlAffineQuadrotor(
            x_lo, x_up, u_lo, u_up, phi_a, phi_b, phi_c, C, u_equilibrium,
            mip_utils.PropagateBoundsMethod.IA, formulation)

    if args.load_lyapunov_relu is None:
        V_lambda = 0.5
        lyapunov_relu = utils.setup_relu((12, 20, 20, 1),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=dtype)
        x_equilibrium = torch.zeros((12, ), dtype=dtype)
        _, S = plant.lqr_control(
            np.diag([1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10.]),
            np.diag([10., 10, 10, 10]), x_equilibrium, u_equilibrium)
        R = torch.from_numpy(S)
    else:
        lyapunov_data = torch.load(args.load_lyapunov_relu)
        lyapunov_relu = utils.setup_relu(
            lyapunov_data["linear_layer_width"],
            params=None,
            negative_slope=lyapunov_data["negative_slope"],
            bias=True,
            dtype=dtype)
        lyapunov_relu.load_state_dict(lyapunov_data["state_dict"])
        V_lambda = lyapunov_data["V_lambda"]
        R = lyapunov_data["R"]

    lyapunov_hybrid_system = control_lyapunov.ControlLyapunov(
        dynamics_model, lyapunov_relu,
        control_lyapunov.SubgradientPolicy(np.array([0.])))

    R_options = r_options.SearchRfreeOptions(R.shape)
    R_options.set_variable_value(R.detach().numpy())

    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, None, x_lo, x_up,
                                        u_lo, u_up)

    dut = train_lyapunov.TrainLyapunovReLU(lyapunov_hybrid_system, V_lambda,
                                           x_equilibrium, R_options)

    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_epsilon = 0.1
    dut.lyapunov_derivative_mip_clamp_min = -1.
    dut.lyapunov_derivative_mip_cost_weight = 1.
    dut.output_flag = True
    dut.learning_rate = 0.003
    dut.max_iterations = args.max_iterations
    dut.lyapunov_derivative_mip_params = {gurobipy.GRB.Param.OutputFlag: True}
    dut.enable_wandb = args.enable_wandb
    state_samples_all = torch.empty((0, 12), dtype=dtype)
    dut.train(state_samples_all)
    pass
