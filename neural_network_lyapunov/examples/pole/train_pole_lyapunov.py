import neural_network_lyapunov.examples.pole.pole_relu_system as \
    pole_relu_system
import neural_network_lyapunov.examples.pole.pole as pole
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.train_lyapunov_barrier as train_lyapunov_barrier
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.train_utils as train_utils

import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train controller/lyapunov for pole/end-effector system.")
    parser.add_argument("--load_forward_model",
                        type=str,
                        default=None,
                        help="path to load dynamics model")
    parser.add_argument("--load_lyapunov_relu",
                        type=str,
                        default=None,
                        help="path to load lyapunov network")
    parser.add_argument("--load_controller_relu",
                        type=str,
                        default=None,
                        help="path to load controller data.")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()

    dtype = torch.float64
    if args.load_forward_model:
        forward_model_data = torch.load(args.load_forward_model)
        forward_model = utils.setup_relu(
            forward_model_data["linear_layer_width"],
            params=None,
            bias=forward_model_data["bias"],
            negative_slope=forward_model_data["negative_slope"],
            dtype=dtype)
        forward_model.load_state_dict(forward_model_data["state_dict"])

    plant = pole.Pole(m_sphere=0.1649, m_ee=0.2, length=0.82)
    dt = 0.01
    x_lo = torch.tensor([-0.1, -0.1, -0.2, -0.2, -0.1, -0.1, -0.1],
                        dtype=dtype)
    x_up = -x_lo
    u_lo = torch.tensor([-1, -1, -0.5], dtype=dtype) * (
        plant.m_sphere + plant.m_ee) * plant.gravity
    u_up = -u_lo
    forward_system = pole_relu_system.PoleReluSystem(
        x_lo, x_up, u_lo, u_up, forward_model, dt,
        (plant.m_ee + plant.m_sphere) * plant.gravity)

    if args.load_lyapunov_relu is None:
        lyapunov_relu = utils.setup_relu((7, 14, 10, 6, 1),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=dtype)
        V_lambda = 0.1
        R = torch.rand((9, 7), dtype=dtype)
    else:
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

    if args.load_controller_relu is None:
        controller_relu = utils.setup_relu((7, 7, 4, 3),
                                           params=None,
                                           negative_slope=0.1,
                                           bias=True,
                                           dtype=dtype)
    else:
        controller_data = torch.load(args.load_controller_relu)
        controller_relu = utils.setup_relu(
            controller_data["linear_layer_width"],
            params=None,
            negative_slope=controller_data["negative_slope"],
            bias=controller_data["bias"],
            dtype=dtype)
        controller_relu.load_state_dict(controller_data["state_dict"])

    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, controller_relu,
                                        x_lo, x_up, u_lo, u_up)

    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy())
    lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(closed_loop_system,
                                                     lyapunov_relu)

    R_options = r_options.SearchRfreeOptions(R.shape)
    R_options.set_variable_value(R.detach().numpy())

    dut = train_lyapunov_barrier.Trainer()
    dut.add_lyapunov(lyap, V_lambda, closed_loop_system.x_equilibrium,
                     R_options)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-5
    dut.lyapunov_positivity_convergence_tol = 1e-5
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.1
    dut.lyapunov_derivative_epsilon = 0.001
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    dut.enable_wandb = args.enable_wandb
    dut.output_flag = True

    dut.train(torch.empty((0, 7), dtype=dtype))

    pass
