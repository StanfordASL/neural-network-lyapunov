"""
Train the quadrotor which directly controls the thrust (NOT using pixhawk
controller).
"""
import neural_network_lyapunov.examples.quadrotor3d.quadrotor as quadrotor
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.examples.quadrotor3d.quadrotor_feedback_system\
    as quadrotor_feedback_system

import torch
import numpy as np
import scipy.integrate
import argparse
import os
import gurobipy


def generate_quadrotor_dynamics_data(dt):
    dtype = torch.float64
    plant = quadrotor.Quadrotor(dtype)

    rpy_angularvel_lo = torch.tensor([
        -np.pi, -0.48 * np.pi, -np.pi, -2. / 3 * np.pi, -2. / 3 * np.pi,
        -2. / 3 * np.pi
    ],
                                     dtype=dtype)
    rpy_angularvel_up = -rpy_angularvel_lo
    u_lo = torch.tensor([0, 0, 0, 0], dtype=dtype)
    u_up = plant.hover_thrust * 3 * torch.ones((4, ), dtype=dtype)
    rpy_angularvel_samples = utils.uniform_sample_in_box(
        rpy_angularvel_lo, rpy_angularvel_up, 2000)
    x_samples = torch.cat(
        (torch.zeros((rpy_angularvel_samples.shape[0], 3),
                     dtype=dtype), rpy_angularvel_samples[:, :3],
         torch.zeros((rpy_angularvel_samples.shape[0], 3),
                     dtype=dtype), rpy_angularvel_samples[:, 3:6]),
        dim=1)
    u_samples = utils.uniform_sample_in_box(u_lo, u_up, 3000)
    xu_tensors = []
    x_next_tensors = []
    for i in range(x_samples.shape[0]):
        for j in range(u_samples.shape[0]):
            result = scipy.integrate.solve_ivp(
                lambda t, x: plant.dynamics(x, u_samples[j, :].detach().numpy(
                )), (0, dt), x_samples[i, :].detach().numpy())
            xu_tensors.append(
                torch.cat((x_samples[i, :], u_samples[j, :])).reshape((1, -1)))
            x_next_tensors.append(
                torch.from_numpy(result.y[:, -1]).reshape((1, -1)))
    dataset_input = torch.cat(xu_tensors, dim=0)
    dataset_output = torch.cat(x_next_tensors, dim=0)
    return torch.utils.data.TensorDataset(dataset_input, dataset_output)


def train_forward_model(forward_model, model_dataset, num_epochs, lr):
    # The forward model maps (rpy[n], angular_vel[n], u[n]) to
    # (rpy[n+1], posdot[n+1] - posdot[n], angular_vel[n+1])
    plant = quadrotor.Quadrotor(torch.float64)
    u_equilibrium = plant.hover_thrust * torch.ones((4, ), dtype=torch.float64)
    xu_inputs, x_next_outputs = model_dataset[:]
    network_input_data = torch.cat(
        (xu_inputs[:, 3:6], xu_inputs[:, 9:12], xu_inputs[:, -4:]), dim=1)
    network_output_data = torch.cat(
        (x_next_outputs[:, 3:6], x_next_outputs[:, 6:9] - xu_inputs[:, 6:9],
         x_next_outputs[:, 9:12]),
        dim=1)
    training_dataset = torch.utils.data.TensorDataset(network_input_data,
                                                      network_output_data)

    def compute_rpy_delta_posdot_angularvel_next(model, network_input):
        return model(network_input) - model(
            torch.cat((torch.zeros((6, ), dtype=torch.float64), u_equilibrium),
                      dim=0))

    utils.train_approximator(training_dataset,
                             forward_model,
                             compute_rpy_delta_posdot_angularvel_next,
                             batch_size=50,
                             num_epochs=num_epochs,
                             lr=lr)


def train_lqr_value_approximator(lyapunov_relu, V_lambda, R, x_equilibrium,
                                 x_lo, x_up, num_samples, lqr_S: torch.Tensor,
                                 num_epochs):
    """
    We train both lyapunov_relu and R such that ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁
    approximates the lqr cost-to-go.
    """
    x_samples = utils.uniform_sample_in_box(x_lo, x_up, num_samples)
    V_samples = torch.sum((x_samples - x_equilibrium) *
                          (lqr_S @ (x_samples - x_equilibrium).T).T,
                          dim=1).reshape((-1, 1))
    state_value_dataset = torch.utils.data.TensorDataset(x_samples, V_samples)
    R.requires_grad_(True)

    def compute_v(model, x):
        return model(x) - model(x_equilibrium) + V_lambda * torch.norm(
            R @ (x - x_equilibrium).T, p=1, dim=0).reshape((-1, 1))

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
    u_samples = -(lqr_K @ (x_samples - x_equilibrium).T).T + u_equilibrium
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
    parser = argparse.ArgumentParser(description="quadrotor training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument("--load_dynamics_data",
                        type=str,
                        default=None,
                        help="path to the dynamics data.")
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--load_forward_model",
                        type=str,
                        default=None,
                        help="path to the forward model.")
    parser.add_argument("--forward_model_hidden_layer_width",
                        "--arg",
                        nargs="+",
                        type=int)
    parser.add_argument("--load_lyapunov_relu",
                        type=str,
                        default=None,
                        help="path to lyapunov relu")
    parser.add_argument("--load_controller_relu",
                        type=str,
                        default=None,
                        help="path to controller relu")
    parser.add_argument("--train_lqr_approximator", action="store_true")
    parser.add_argument("--train_on_samples", action="store_true")
    parser.add_argument("--search_R", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--pretrain_num_epochs", type=int, default=20)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--train_adversarial", action="store_true")
    parser.add_argument(
        "--training_set",
        type=str,
        default=None,
        help="path to the training set for adversarial training.")
    parser.add_argument(
        "--derivative_mip_num_strengthen_pts", type=int, default=0)
    parser.add_argument(
        "--derivative_mip_strengthen_binary", action="store_true")
    args = parser.parse_args()
    dt = 0.01
    dtype = torch.float64
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.generate_dynamics_data:
        model_dataset = generate_quadrotor_dynamics_data(dt)

    if args.train_forward_model:
        if args.load_dynamics_data:
            model_dataset = torch.load(args.load_dynamics_data)
        else:
            model_dataset = torch.load(
                dir_path + "/data/quadrotor_forward_dynamics_dataset_4.pt")
        forward_model_linear_layer_width = [
            10
        ] + args.forward_model_hidden_layer_width + [9]
        forward_model = utils.setup_relu(
            tuple(forward_model_linear_layer_width),
            params=None,
            bias=True,
            negative_slope=0.1,
            dtype=dtype)
        train_forward_model(forward_model,
                            model_dataset,
                            num_epochs=100,
                            lr=0.001)

    if args.load_forward_model:
        forward_model = torch.load(args.load_forward_model)

    lyapunov_relu = utils.setup_relu((12, 14, 10, 1),
                                     params=None,
                                     negative_slope=0.1,
                                     bias=True,
                                     dtype=dtype)
    V_lambda = 0.6
    R = torch.zeros((15, 12), dtype=dtype)
    R[:12, :] = torch.eye(12, dtype=dtype)
    R[12, 0] = 1.
    R[12, 1] = -1.
    R[13, 2] = 1.
    R[13, 3] = -1.
    R[14, 4] = 1.
    R[14, 5] = -1.
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

    controller_relu = utils.setup_relu((12, 12, 8, 4),
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

    x_equilibrium = torch.zeros((12, ), dtype=dtype)
    plant = quadrotor.Quadrotor(dtype)
    u_equilibrium = plant.hover_thrust * torch.ones((4, ), dtype=dtype)
    x_lo = torch.tensor([
        -0.04, -0.04, -0.04, -0.04 * np.pi, -0.04 * np.pi, -0.04 * np.pi, -0.2,
        -0.2, -0.4, -np.pi * 0.08, -np.pi * 0.08, -np.pi * 0.08
    ],
                        dtype=dtype) * 0.5
    x_up = -x_lo
    u_lo = torch.zeros((4, ), dtype=dtype)
    u_up = 3 * u_equilibrium

    plant_A, plant_B = plant.dynamics_gradient(x_equilibrium.detach().numpy(),
                                               u_equilibrium.detach().numpy())
    lqr_Q = np.diag([10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1.])
    lqr_R = np.eye(4)
    lqr_S = scipy.linalg.solve_continuous_are(plant_A, plant_B, lqr_Q, lqr_R)
    lqr_K = np.linalg.solve(lqr_R, plant_B.T @ lqr_S)

    if args.train_lqr_approximator:
        train_lqr_control_approximator(controller_relu,
                                       x_equilibrium,
                                       u_equilibrium,
                                       x_lo,
                                       x_up,
                                       100000,
                                       torch.from_numpy(lqr_K),
                                       num_epochs=100)
        train_lqr_value_approximator(lyapunov_relu,
                                     V_lambda,
                                     R,
                                     x_equilibrium,
                                     x_lo,
                                     x_up,
                                     100000,
                                     torch.from_numpy(lqr_S),
                                     num_epochs=100)

    forward_system = quadrotor.QuadrotorReLUSystem(dtype, x_lo, x_up, u_lo,
                                                   u_up, forward_model,
                                                   plant.hover_thrust, dt)
    if args.train_adversarial:
        forward_system.network_bound_propagate_method =\
            mip_utils.PropagateBoundsMethod.MIP
    closed_loop_system = quadrotor_feedback_system.QuadrotorFeedbackSystem(
        forward_system, controller_relu,
        u_lo.detach().numpy(),
        u_up.detach().numpy())
    if args.train_adversarial:
        closed_loop_system.controller_network_bound_propagate_method =\
            mip_utils.PropagateBoundsMethod.MIP
    lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(closed_loop_system,
                                                     lyapunov_relu)
    if args.train_adversarial:
        lyap.network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.MIP

    if args.search_R:
        _, R_sigma, _ = np.linalg.svd(R.detach().numpy())
        R_options = r_options.SearchRwithSVDOptions(R.shape, R_sigma * 0.8)
        R_options.set_variable_value(R.detach().numpy())
    else:
        R_options = r_options.FixedROptions(R)
    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, controller_relu,
                                        x_lo, x_up, u_lo, u_up)

    dut = train_lyapunov.TrainLyapunovReLU(lyap, V_lambda,
                                           closed_loop_system.x_equilibrium,
                                           R_options)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-5
    dut.lyapunov_positivity_convergence_tol = 5E-6
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.1
    dut.lyapunov_derivative_epsilon = 0.001
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    dut.lyapunov_derivative_mip_params = {
        # gurobipy.GRB.Attr.MIPGap: 1.,
        gurobipy.GRB.Param.OutputFlag: True,
        # gurobipy.GRB.Param.TimeLimit: 900,
        gurobipy.GRB.Param.MIPFocus: 1
    }
    state_samples_all = utils.uniform_sample_in_box(x_lo, x_up, 10000)
    dut.output_flag = True
    dut.derivative_mip_num_strengthen_pts = \
        args.derivative_mip_num_strengthen_pts
    if args.train_on_samples:
        dut.train_lyapunov_on_samples(state_samples_all,
                                      num_epochs=args.pretrain_num_epochs,
                                      batch_size=50)
    dut.enable_wandb = args.enable_wandb
    dut.derivative_mip_strengthen_binary = \
        args.derivative_mip_strengthen_binary
    if args.train_adversarial:
        options = train_lyapunov.TrainLyapunovReLU.AdversarialTrainingOptions()
        options.num_batches = 10
        options.num_epochs_per_mip = 30
        options.positivity_samples_pool_size = 50000
        options.derivative_samples_pool_size = 500000
        options.adversarial_cluster_radius = 1E-5
        dut.lyapunov_positivity_mip_pool_solutions = 1000
        dut.lyapunov_derivative_mip_pool_solutions = 20000
        dut.add_positivity_adversarial_state = True
        dut.add_derivative_adversarial_state = True
        if args.training_set:
            training_set = torch.load(args.training_set)
            positivity_state_samples_init = training_set[
                "positivity_state_samples_all"]
            derivative_state_samples_init = training_set[
                "derivative_state_samples_all"]
        else:
            positivity_state_samples_init = utils.uniform_sample_in_box(
                x_lo, x_up, 500)
            derivative_state_samples_init = positivity_state_samples_init
        result = dut.train_adversarial(positivity_state_samples_init,
                                       derivative_state_samples_init, options)
    else:
        dut.train(torch.empty((0, 12), dtype=dtype))
    pass
