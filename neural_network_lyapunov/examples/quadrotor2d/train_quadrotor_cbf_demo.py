import neural_network_lyapunov.train_barrier as train_barrier
import neural_network_lyapunov.control_barrier as control_barrier
import neural_network_lyapunov.examples.quadrotor2d.control_affine_quadrotor\
    as control_affine_quadrotor
import neural_network_lyapunov.examples.quadrotor2d.quadrotor_2d as \
    quadrotor_2d
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.mip_utils as mip_utils

import torch
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="quadrotor2d cbf training demo")
    parser.add_argument("--load_forward_model",
                        type=str,
                        default=None,
                        help="path to load the forward model")
    parser.add_argument("--load_barrier_relu",
                        type=str,
                        default=None,
                        help="path to load the control barrier model")
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--enable_wandb", action="store_true")
    args = parser.parse_args()
    dtype = torch.float64

    plant = quadrotor_2d.Quadrotor2D(dtype)
    x_lo = torch.tensor([-0.5, -0.5, -0.3 * np.pi, -3, -3, -1.5], dtype=dtype)
    x_up = -x_lo
    u_lo = torch.tensor([0, 0], dtype=dtype)
    u_up = torch.tensor([1, 1], dtype=dtype) * plant.mass * plant.gravity * 1.5
    dynamics_model_data = torch.load(args.load_forward_model)
    phi_b = utils.setup_relu(
        dynamics_model_data["phi_b"]["linear_layer_width"],
        params=None,
        negative_slope=dynamics_model_data["phi_b"]["negative_slope"],
        bias=dynamics_model_data["phi_b"]["bias"],
        dtype=dtype)
    phi_b.load_state_dict(dynamics_model_data["phi_b"]["state_dict"])
    u_equilibrium = torch.tensor([0.5, 0.5],
                                 dtype=dtype) * plant.mass * plant.gravity
    dynamics_model = control_affine_quadrotor.ControlAffineQuadrotor2d(
        x_lo,
        x_up,
        u_lo,
        u_up,
        phi_b,
        u_equilibrium,
        method=mip_utils.PropagateBoundsMethod.IA)
    x_equilibrium = torch.zeros((6,), dtype=dtype)

    if args.load_barrier_relu is None:
        barrier_relu = utils.setup_relu((6, 15, 15, 1),
                                        params=None,
                                        negative_slope=0.1,
                                        bias=True,
                                        dtype=dtype)
        c = 0.5
        x_star = x_equilibrium
        barrier_system = control_barrier.ControlBarrier(
            dynamics_model, barrier_relu)
    else:
        barrier_data = torch.load(args.load_barrier_relu)
        barrier_relu = utils.setup_relu(
            barrier_data["linear_layer_width"],
            params=None,
            negative_slope=barrier_data["negative_slope"],
            bias=True)
        barrier_relu.load_state_dict(barrier_data["state_dict"])
        x_star = barrier_data["x_star"]
        c = barrier_data["c"]
        barrier_system = control_barrier.ControlBarrier(
            dynamics_model, barrier_relu)

    # The unsafe region is z < -0.2
    unsafe_region_cnstr = gurobi_torch_mip.MixedIntegerConstraintsReturn()
    unsafe_region_cnstr.Ain_input = torch.tensor([[0, 1, 0, 0, 0, 0]],
                                                 dtype=dtype)
    unsafe_region_cnstr.rhs_in = torch.tensor([-.2], dtype=dtype)

    verify_region_boundary = utils.box_boundary(x_lo, x_up)

    epsilon = 0.1

    dut = train_barrier.TrainBarrier(barrier_system, x_star, c,
                                     unsafe_region_cnstr,
                                     verify_region_boundary, epsilon)
    dut.max_iterations = args.max_iterations
    dut.enable_wandb = args.enable_wandb

    dut.train(torch.zeros((0, 6), dtype=dtype))
    pass
