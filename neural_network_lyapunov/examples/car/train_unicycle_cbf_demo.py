import neural_network_lyapunov.examples.car.control_affine_unicycle as \
    control_affine_unicycle
import neural_network_lyapunov.control_barrier as control_barrier
import neural_network_lyapunov.barrier as barrier
import neural_network_lyapunov.train_barrier as train_barrier
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import torch
import argparse
import numpy as np


def train_forward_model(
        dynamics_model: control_affine_unicycle.ControlAffineUnicycle):
    dtype = torch.float64
    theta = torch.linspace(-1.5 * np.pi, 1.5 * np.pi, 500,
                           dtype=dtype).unsqueeze(1)
    phi_out = torch.cat((torch.cos(theta), torch.sin(theta)), dim=1)
    v_dataset = torch.utils.data.TensorDataset(theta, phi_out)

    def compute_phi(phi, theta):
        return phi(theta)

    utils.train_approximator(v_dataset,
                             dynamics_model.phi,
                             compute_phi,
                             batch_size=20,
                             num_epochs=500,
                             lr=0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unicycle cbf training")
    parser.add_argument("--load_dynamics_data",
                        type=str,
                        default=None,
                        help="path to load dynamics data")
    parser.add_argument("--train_forward_model",
                        type=str,
                        default=None,
                        help="path to save trained forward_model")
    parser.add_argument("--load_forward_model",
                        type=str,
                        default=None,
                        help="path to load the forward model")
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--enable_wandb", action="store_true")
    args = parser.parse_args()
    dtype = torch.float64

    u_lo = torch.tensor([0, -np.pi / 4], dtype=dtype)
    u_up = torch.tensor([1, np.pi / 4], dtype=dtype)

    x_lo = torch.tensor([-1, -1, -1.5 * np.pi], dtype=dtype)
    x_up = -x_lo

    if args.train_forward_model is not None:
        phi = utils.setup_relu((1, 10, 10, 2),
                               params=None,
                               negative_slope=0.1,
                               bias=True,
                               dtype=dtype)
        dynamics_model = control_affine_unicycle.ControlAffineUnicycleApprox(
            phi, x_lo, x_up, u_lo, u_up, mip_utils.PropagateBoundsMethod.IA)
        train_forward_model(dynamics_model)
        linear_layer_width, negative_slope, bias = \
            utils.extract_relu_structure(phi)
        torch.save(
            {
                "linear_layer_width": linear_layer_width,
                "state_dict": phi.state_dict(),
                "negative_slope": negative_slope,
                "bias": bias,
                "x_lo": x_lo,
                "x_up": x_up,
                "u_lo": u_lo,
                u_up: "u_up"
            }, args.train_forward_model)
    elif args.load_forward_model is not None:
        dynamics_model_data = torch.load(args.load_forward_model)
        phi = utils.setup_relu(
            dynamics_model_data["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["negative_slope"],
            bias=dynamics_model_data["bias"],
            dtype=dtype)
        phi.load_state_dict(dynamics_model_data["state_dict"])
        dynamics_model = control_affine_unicycle.ControlAffineUnicycleApprox(
            phi, x_lo, x_up, u_lo, u_up, mip_utils.PropagateBoundsMethod.IA)

    barrier_relu = utils.setup_relu((3, 8, 4, 1),
                                    params=None,
                                    negative_slope=0.1,
                                    bias=True)
    x_star = torch.tensor([0, 0, 0], dtype=dtype)
    c = 0.5
    barrier_system = control_barrier.ControlBarrier(dynamics_model,
                                                    barrier_relu)
    verify_region_boundary = utils.box_boundary(x_lo, x_up)
    unsafe_region_cnstr = gurobi_torch_mip.MixedIntegerConstraintsReturn()
    epsilon = 0.2
    inf_norm_term = barrier.InfNormTerm.from_bounding_box(x_lo, x_up, 0.7)

    dut = train_barrier.TrainBarrier(barrier_system, x_star, c,
                                     unsafe_region_cnstr,
                                     verify_region_boundary, epsilon,
                                     inf_norm_term)
    dut.enable_wandb = args.enable_wandb

    unsafe_state_samples = torch.zeros((0, 3), dtype=dtype)
    boundary_state_samples = torch.zeros((0, 3), dtype=dtype)
    deriv_state_samples = torch.zeros((0, 3), dtype=dtype)

    is_success = dut.train(unsafe_state_samples, boundary_state_samples,
                           deriv_state_samples)
    pass
