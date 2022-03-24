import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.continuous_time_lyapunov as \
    continuous_time_lyapunov
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.examples.quadrotor2d.quadrotor_2d as\
    quadrotor_2d

import torch
import numpy as np
import argparse
import gurobipy


def generate_dynamics_data():
    dtype = torch.float64
    plant = quadrotor_2d.Quadrotor2D(dtype)
    x_lo = torch.tensor([0, 0, -np.pi * 0.7, 0, 0, 0], dtype=dtype)
    x_up = -x_lo
    u_lo = torch.tensor([0, 0], dtype=dtype)
    u_up = plant.mass * plant.gravity / 2 * 3 * torch.tensor([1, 1],
                                                             dtype=dtype)
    xu_samples = utils.uniform_sample_in_box(torch.cat((x_lo, u_lo)),
                                             torch.cat((x_up, u_up)), 100000)
    xdot = torch.empty((xu_samples.shape[0], 6), dtype=dtype)
    with torch.no_grad():
        for i in range(xu_samples.shape[0]):
            xdot[i] = plant.dynamics(xu_samples[i, :6], xu_samples[i, 6:])
    return torch.utils.data.TensorDataset(xu_samples, xdot)


def train_forward_model(dynamics_relu, model_dataset, u_equilibrium):
    xu_inputs, xdot_outputs = model_dataset[:]
    qddot_dataset = torch.utils.data.TensorDataset(
        torch.cat((xu_inputs[:, 2:3], xu_inputs[:, 6:]), dim=1),
        xdot_outputs[:, 3:])

    def compute_qddot(model, state_action):
        return model(state_action) - model(
            torch.cat((torch.tensor([0], dtype=torch.float64), u_equilibrium)))

    utils.train_approximator(qddot_dataset,
                             dynamics_relu,
                             compute_qddot,
                             batch_size=50,
                             num_epochs=200,
                             lr=0.01)
    pass


def train_lqr_value_approximator(lyapunov_relu, V_lambda, R, x_equilibrium,
                                 x_lo, x_up, num_samples, lqr_S: torch.Tensor):
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
                             num_epochs=200,
                             lr=0.001,
                             additional_variable=[R])
    R.requires_grad_(False)


def train_lqr_control_approximator(controller_relu, x_equilibrium,
                                   u_equilibrium, x_lo, x_up, num_samples,
                                   lqr_K: torch.Tensor):
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
                             num_epochs=50,
                             lr=0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D quadrotor training demo")
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
    parser.add_argument("--train_lqr_approximator", action="store_true")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--max_iterations",
                        type=int,
                        default=500,
                        help="number of iterations in training Lyapunov")
    parser.add_argument("--train_adversarial", action="store_true")
    parser.add_argument("--training_set", type=str, default=None)

    args = parser.parse_args()

    dtype = torch.float64
    plant = quadrotor_2d.Quadrotor2D(dtype)

    if args.generate_dynamics_data:
        model_dataset = generate_dynamics_data()
        torch.save(model_dataset, args.generate_dynamics_data)
    if args.load_dynamics_data:
        model_dataset = torch.load(args.load_dynamics_data)

    dynamics_relu = utils.setup_relu((3, 10, 10, 3),
                                     params=None,
                                     negative_slope=0.01,
                                     bias=True,
                                     dtype=dtype)

    u_equilibrium = plant.mass * plant.gravity / 2 * torch.tensor([1, 1],
                                                                  dtype=dtype)

    if args.train_forward_model is not None:
        train_forward_model(dynamics_relu, model_dataset, u_equilibrium)
        linear_layer_width, negative_slope, bias = \
            utils.extract_relu_structure(dynamics_relu)
        torch.save(
            {
                "linear_layer_width": linear_layer_width,
                "state_dict": dynamics_relu.state_dict(),
                "negative_slope": negative_slope,
                "bias": bias
            }, args.train_forward_model)

    if args.load_forward_model is not None:
        dynamics_model_data = torch.load(args.load_forward_model)
        dynamics_relu = utils.setup_relu(
            dynamics_model_data["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["negative_slope"],
            bias=dynamics_model_data["bias"],
            dtype=dtype)
        dynamics_relu.load_state_dict(dynamics_model_data["state_dict"])

    x_lo = 0.1 * torch.tensor(
        [-0.1, -0.1, -0.1 * np.pi, -0.3, -0.3, -0.3 * np.pi], dtype=dtype)
    x_up = -x_lo
    u_lo = torch.tensor([0, 0], dtype=dtype)
    u_up = torch.tensor([6, 6], dtype=dtype)
    dynamics_model = quadrotor_2d.QuadrotorReluContinuousTime(
        dtype, x_lo, x_up, u_lo, u_up, dynamics_relu, u_equilibrium)

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
        controller_relu = utils.setup_relu((6, 6, 4, 2),
                                           params=None,
                                           negative_slope=0.1,
                                           bias=True,
                                           dtype=dtype)
    lqr_Q = np.diag([10, 10, 10, 1, 1, plant.length / 2. / np.pi])
    lqr_R = np.array([[0.1, 0.05], [0.05, 0.1]])
    x_star = np.zeros((6, ))
    u_star = plant.u_equilibrium.detach().numpy()
    K, S = plant.lqr_control(lqr_Q, lqr_R, x_star, u_star)
    S_eig_value, S_eig_vec = np.linalg.eig(S)
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
        lyapunov_relu = utils.setup_relu((6, 10, 10, 4, 1),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=dtype)
        V_lambda = 0.5
        R = torch.from_numpy(S) + 0.01 * torch.eye(6, dtype=dtype)
    if args.train_lqr_approximator:
        x_equilibrium = torch.zeros((6, ), dtype=dtype)
        train_lqr_control_approximator(controller_relu, x_equilibrium,
                                       u_equilibrium, x_lo, x_up, 100000,
                                       torch.from_numpy(K))
        train_lqr_value_approximator(lyapunov_relu, V_lambda, R, x_equilibrium,
                                     x_lo, x_up, 100000, torch.from_numpy(S))
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

    dut = train_lyapunov.TrainLyapunovReLU(lyapunov_hybrid_system, V_lambda,
                                           closed_loop_system.x_equilibrium,
                                           R_options)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-5
    dut.lyapunov_positivity_convergence_tol = 1E-5
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.2
    dut.lyapunov_derivative_epsilon = 0.1
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    dut.output_flag = True
    dut.enable_wandb = args.enable_wandb
    if args.train_adversarial:
        options = train_lyapunov.TrainLyapunovReLU.AdversarialTrainingOptions()
        options.num_batches = 10
        options.num_epochs_per_mip = 30
        options.perturb_derivative_sample_count = 5
        options.perturb_derivative_sample_std = 1E-4
        options.positivity_samples_pool_size = 10000
        options.derivative_samples_pool_size = 100000
        options.adversarial_cluster_radius = 1E-5
        dut.lyapunov_positivity_mip_pool_solutions = 100
        dut.lyapunov_derivative_mip_pool_solutions = 500
        dut.add_derivative_adversarial_state = True
        dut.add_positivity_adversarial_state = True
        dynamics_model.network_bound_propagate_method =\
            mip_utils.PropagateBoundsMethod.MIP
        dut.lyapunov_hybrid_system.network_bound_propagate_method =\
            mip_utils.PropagateBoundsMethod.MIP
        closed_loop_system.controller_network_bound_propagate_method =\
            mip_utils.PropagateBoundsMethod.MIP
        dut.lyapunov_derivative_mip_params = {
            gurobipy.GRB.Param.OutputFlag: False
        }
        dut.sample_loss_reduction = "4norm"
        dut.learning_rate = 0.01
        if args.training_set:
            training_set_data = torch.load(args.training_set)
            positivity_state_samples_init = training_set_data[
                "positivity_state_samples"]
            derivative_state_samples_init = training_set_data[
                "derivative_state_samples"]
        else:
            positivity_state_samples_init = utils.uniform_sample_in_box(
                x_lo, x_up, 1000)
            derivative_state_samples_init = positivity_state_samples_init
        result = dut.train_adversarial(positivity_state_samples_init,
                                       derivative_state_samples_init, options)
    else:
        dut.train(torch.empty((0, 6), dtype=dtype))
    pass
