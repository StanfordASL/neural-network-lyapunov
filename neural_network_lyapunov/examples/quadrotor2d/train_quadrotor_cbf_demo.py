import neural_network_lyapunov.train_barrier as train_barrier
import neural_network_lyapunov.barrier as barrier
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
import gurobipy
import scipy.integrate
import matplotlib.pyplot as plt


def simulate(dynamics_model: control_affine_quadrotor.ControlAffineQuadrotor2d,
             barrier_relu, x_star, c, inf_norm_term, u_lo, u_up, epsilon, x0,
             pos_des, T):
    """
    Simulate the system to go to a desired hovering position, while respecting
    the barrier cerificate.
    """
    dtype = torch.float64
    plant = quadrotor_2d.Quadrotor2D(dtype)
    dut = control_barrier.ControlBarrier(dynamics_model, barrier_relu)

    x_des = np.array([pos_des[0], pos_des[1], 0, 0, 0, 0])
    h_des = barrier_relu(torch.from_numpy(x_des)) - barrier_relu(x_star) + c
    if h_des < 0:
        raise Exception(f"The desired state has h value = {h_des.item()} < 0")
    u_des = np.array([1, 1]) * plant.mass * plant.gravity * 0.5

    A, B = plant.linearized_dynamics(x_des, u_des)
    Q = np.diag([1, 1, 1, 10, 10, 10])
    R = np.diag([1, 1.])
    K, S = plant.lqr_control(Q, R, x_des, u_des)

    def compute_control(x):
        prog = gurobipy.Model()
        u = prog.addVars(2, lb=u_lo.tolist(), ub=u_up.tolist())
        u_var = [u[0], u[1]]
        x_torch = torch.from_numpy(x)
        with torch.no_grad():
            dhdx = dut._barrier_gradient(x_torch, inf_norm_term)

            f = dynamics_model.f(x_torch)
            G = dynamics_model.G(x_torch)
            h = dut.barrier_value(x_torch,
                                  x_star,
                                  c,
                                  inf_norm_term=inf_norm_term)
            for i in range(dhdx.shape[0]):
                prog.addLConstr(gurobipy.LinExpr((dhdx[i] @ G).tolist(),
                                                 u_var),
                                sense=gurobipy.GRB.GREATER_EQUAL,
                                rhs=(-epsilon * h - dhdx[i] @ f).item())
            # Add the cost
            # min (u-u_des)ᵀR(u-u_des) + 2(x-x_des)ᵀS(A(x-x_des)+Bu)
            cost = gurobipy.QuadExpr()
            cost.add((u[0] - u_des[0]) * (u[0] - u_des[0]) * R[0, 0] +
                     (u[1] - u_des[1]) * (u[1] - u_des[1]) * R[1, 1] + 2 *
                     (u[0] - u_des[0]) * (u[1] - u_des[1]) * R[0, 1])
            cost.addTerms(2 * (x - x_des) @ S @ B, u_var)
            prog.setObjective(cost, sense=gurobipy.GRB.MINIMIZE)
            prog.setParam(gurobipy.GRB.Param.OutputFlag, False)
            prog.optimize()
            assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
            u_val = np.array([v.x for v in u_var])
            return u_val

    def plant_dynamics(t, x):
        u = compute_control(x)
        return plant.dynamics(x, u)

    def nn_plant_dynamics(t, x):
        u = compute_control(x)
        with torch.no_grad():
            return dynamics_model.dynamics(torch.from_numpy(x),
                                           torch.from_numpy(u))

    def reach_goal(t, x):
        return np.linalg.norm(x - x_des) - 1E-3

    reach_goal.terminal = True

    def exit_boundary(t, x):
        return np.minimum((x_up.detach().numpy() + 0.01 - x).min(),
                          (x - x_lo.detach().numpy() + 0.01).min())

    exit_boundary.terminal = True

    result_newton = scipy.integrate.solve_ivp(
        plant_dynamics, [0, T],
        x0,
        events=[reach_goal, exit_boundary],
        max_step=0.01)
    result_nn = scipy.integrate.solve_ivp(nn_plant_dynamics, [0, T],
                                          x0,
                                          events=[reach_goal, exit_boundary],
                                          max_step=0.001)

    def plot_result(result):
        nT = result.t.shape[0]
        u_val = np.zeros((2, nT))
        hdot = np.zeros((nT, ))
        for i in range(nT):
            u_val[:, i] = compute_control(result.y[:, i])
            hdot[i] = dut.minimal_barrier_derivative_given_action(
                torch.from_numpy(result.y[:, i]),
                torch.from_numpy(u_val[:, i]),
                inf_norm_term=inf_norm_term).item()
        h_val = dut.barrier_value(
            torch.from_numpy(result.y.T),
            x_star,
            c,
            inf_norm_term=inf_norm_term).detach().numpy()
        fig_u = plt.figure()
        ax_u0 = fig_u.add_subplot(211)
        ax_u0.plot(result.t, u_val[0, :])
        ax_u0.set_title("u")
        ax_u1 = fig_u.add_subplot(212)
        ax_u1.plot(result.t, u_val[1, :])
        ax_u1.set_xlabel("time (s)")

        fig_h = plt.figure()
        ax_h0 = fig_h.add_subplot(211)
        ax_h0.plot(result.t, h_val)
        ax_h0.set_title("h")
        ax_h1 = fig_h.add_subplot(212)
        ax_h1.plot(result.t, hdot)
        ax_h1.set_xlabel("time (s)")

        return fig_u, fig_h, u_val, h_val, hdot

    return result_newton, result_nn


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
    parser.add_argument("--train_on_samples", action="store_true")
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
    x_equilibrium = torch.zeros((6, ), dtype=dtype)

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
    unsafe_height = -0.2
    # unsafe_region_cnstr.Ain_input = torch.tensor([[0, 1, 0, 0, 0, 0]],
    #                                              dtype=dtype)
    # unsafe_region_cnstr.rhs_in = torch.tensor([unsafe_height], dtype=dtype)

    verify_region_boundary = utils.box_boundary(x_lo, x_up)

    epsilon = 0.1

    inf_norm_term = barrier.InfNormTerm(torch.diag(2. / (x_up - x_lo)),
                                        (x_up + x_lo) / (x_up - x_lo))

    dut = train_barrier.TrainBarrier(barrier_system, x_star, c,
                                     unsafe_region_cnstr,
                                     verify_region_boundary, epsilon,
                                     inf_norm_term)
    dut.max_iterations = args.max_iterations
    dut.enable_wandb = args.enable_wandb

    simulate(dynamics_model, barrier_relu, x_star, c, inf_norm_term, u_lo,
             u_up, epsilon, np.zeros((6, )), np.array([0, -0.35]), 3)

    if args.train_on_samples:
        # First train on samples without solving MIP.
        dut.derivative_state_samples_weight = 1.
        dut.boundary_state_samples_weight = 1.
        dut.unsafe_state_samples_weight = 1.
        x_up_unsafe = x_up.clone()
        x_up_unsafe[1] = unsafe_height
        # unsafe_state_samples = utils.uniform_sample_in_box(
        #     x_lo, x_up_unsafe, 1000)
        unsafe_state_samples = torch.empty((0, 6), dtype=dtype)
        boundary_state_samples = utils.uniform_sample_on_box_boundary(
            x_lo, x_up, 3000)
        deriv_state_samples = utils.uniform_sample_in_box(x_lo, x_up, 2000)
        dut.train_on_samples(unsafe_state_samples, boundary_state_samples,
                             deriv_state_samples)
        pass
    unsafe_state_samples = torch.zeros((0, 6), dtype=dtype)
    boundary_state_samples = torch.zeros((0, 6), dtype=dtype)
    deriv_state_samples = torch.zeros((0, 6), dtype=dtype)
    dut.deriv_mip_margin = 0.5
    # dut.verify_region_boundary_mip_cost_weight = 5
    dut.train(unsafe_state_samples, boundary_state_samples,
              deriv_state_samples)
    pass
