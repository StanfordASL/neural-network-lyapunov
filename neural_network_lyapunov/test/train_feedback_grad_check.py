# I suspect there are gradient error when I synthesize the stabilizing
# controller. This file checks the gradient through numeric differentiation.
import torch

import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.test.feedback_gradient_check as\
    feedback_gradient_check
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.feedback_system as feedback_system
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse grad check")
    parser.add_argument(
        "--forward_network", type=str, help="path of the forward relu.")
    parser.add_argument(
        "--controller_network", type=str, help="controller model path")
    parser.add_argument(
        "--lyapunov_network", type=str, help="lyapunov model path")
    args = parser.parse_args()
    # First load the forward system, the controller, and the Lyapunov
    forward_network = torch.load(args.forward_network)
    forward_relu = utils.setup_relu(
        forward_network["linear_layer_width"], params=None,
        negative_slope=forward_network["negative_slope"], bias=True,
        dtype=torch.float64)
    forward_relu.load_state_dict(forward_network["state_dict"])
    q_equilibrium = forward_network["q_equilibrium"]
    u_equilibrium = forward_network["u_equilibrium"]
    dt = forward_network["dt"]

    controller_network = torch.load(args.controller_network)
    controller_relu = utils.setup_relu(
        controller_network["linear_layer_width"], params=None,
        negative_slope=controller_network["negative_slope"], bias=True,
        dtype=torch.float64)
    controller_relu.load_state_dict(controller_network["state_dict"])
    x_lo = controller_network["x_lo"]
    x_up = controller_network["x_up"]
    u_lo = controller_network["u_lo"]
    u_up = controller_network["u_up"]

    lyapunov_network = torch.load(args.lyapunov_network)
    lyapunov_relu = utils.setup_relu(
        lyapunov_network["linear_layer_width"], params=None,
        negative_slope=lyapunov_network["negative_slope"], bias=True,
        dtype=torch.float64)
    lyapunov_relu.load_state_dict(lyapunov_network["state_dict"])
    V_lambda = lyapunov_network["V_lambda"]
    forward_system = relu_system.ReLUSecondOrderSystemGivenEquilibrium(
        torch.float64, x_lo, x_up, u_lo, u_up, forward_relu, q_equilibrium,
        u_equilibrium, dt)
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium, u_lo.detach().numpy(),
        u_up.detach().numpy())
    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
        closed_loop_system, lyapunov_relu)

    x_samples = utils.get_meshgrid_samples(x_lo, x_up, (5, 5), torch.float64)

    # Check sample loss.
    feedback_gradient_check.check_sample_loss_grad(
        lyapunov_hybrid_system, V_lambda, forward_system.x_equilibrium,
        x_samples, atol=1E-5, rtol=1E-5)
    # Check gradient of positivity MIP cost.
    feedback_gradient_check.check_lyapunov_mip_loss_grad(
        lyapunov_hybrid_system, forward_system.x_equilibrium, V_lambda, 0.1,
        True, atol=1E-5, rtol=1E-5)
    # Check gradient of derivative MIP cost.
    feedback_gradient_check.check_lyapunov_mip_loss_grad(
        lyapunov_hybrid_system, forward_system.x_equilibrium, V_lambda, 0.01,
        False, atol=1E-5, rtol=1E-5)
