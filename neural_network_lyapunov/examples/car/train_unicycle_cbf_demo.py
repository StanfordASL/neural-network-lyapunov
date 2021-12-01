import neural_network_lyapunov.examples.car.control_affine_unicycle as \
    control_affine_unicycle
import neural_network_lyapunov.control_barrier as control_barrier
import neural_network_lyapunov.barrier as barrier
import neural_network_lyapunov.train_barrier as train_barrier
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.integrator as integrator
import neural_network_lyapunov.nominal_controller as nominal_controller

import torch
import argparse
import numpy as np
import gurobipy
import matplotlib.pyplot as plt


def simulate(barrier_syatem, dt, nT, x0, u0, x_star, c, epsilon,
             inf_norm_term):
    def compute_control(x, u_des):
        x_torch = torch.from_numpy(x)
        prog = gurobipy.Model()
        u = prog.addVars(2,
                         lb=barrier_system.system.u_lo.tolist(),
                         ub=barrier_system.system.u_up.tolist())
        f = barrier_system.system.f(x_torch).detach().numpy()
        G = barrier_system.system.G(x_torch).detach().numpy()
        dhdx = barrier_system._barrier_gradient(x_torch,
                                                inf_norm_term,
                                                zero_tol=0.).detach().numpy()
        h = barrier_system.barrier_value(
            x_torch, x_star, c, inf_norm_term=inf_norm_term).detach().numpy()
        prog.addMConstrs(dhdx @ G, [u[0], u[1]],
                         sense=gurobipy.GRB.GREATER_EQUAL,
                         b=-epsilon * h - dhdx @ f)
        prog.setObjective((u[0] - u_des[0]) * (u[0] - u_des[0]) +
                          (u[1] - u_des[1]) * (u[1] - u_des[1]),
                          sense=gurobipy.GRB.MINIMIZE)
        prog.setParam(gurobipy.GRB.Param.OutputFlag, False)
        prog.optimize()
        assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
        u_val = np.array([u[0].x, u[1].x])
        return u_val

    x_val = np.zeros((3, nT))
    u_val = np.zeros((2, nT))
    x_val[:, 0] = x0
    for i in range(nT - 1):
        if i == 0:
            u_des = u0
        else:
            u_des = u_val[:, i - 1]
        x_val[:, i + 1], u_val[:, i] = integrator.rk4_constant_control(
            lambda x, u: barrier_system.system.dynamics(
                torch.from_numpy(x), torch.from_numpy(u)).detach().numpy(),
            lambda x: compute_control(x, u_des),
            x_val[:, i],
            dt,
            constant_control_steps=1)
    u_val[:, -1] = compute_control(x_val[:, -1], u_val[:, -2])
    h_val = barrier_system.barrier_value(
        torch.from_numpy(x_val.T), x_star, c,
        inf_norm_term=inf_norm_term).detach().numpy()
    hdot_val = barrier_system.barrier_derivative_given_action_batch(
        torch.from_numpy(x_val.T),
        torch.from_numpy(u_val.T),
        create_graph=False,
        inf_norm_term=inf_norm_term).detach().numpy()

    t_val = dt * np.arange(nT)

    fig_u = plt.figure()
    ax_vel = fig_u.add_subplot(211)
    ax_vel.plot(t_val, u_val[0, :])
    ax_vel.set_ylabel("vel (m/s)")
    ax_thetadot = fig_u.add_subplot(212)
    ax_thetadot.plot(t_val, u_val[1, :])
    ax_thetadot.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax_thetadot.set_xlabel("t (s)")

    fig_h = plt.figure()
    ax_h = fig_h.add_subplot(211)
    ax_h.plot(t_val, h_val)
    ax_h.set_ylabel("h")
    ax_hdot = fig_h.add_subplot(212)
    ax_hdot.plot(t_val, hdot_val)
    ax_hdot.set_ylabel(r"$\dot{h}$")
    ax_hdot.set_xlabel("t (s)")

    fig_x = plt.figure()
    ax_x = fig_x.add_subplot(311)
    ax_x.plot(t_val, x_val[0, :])
    ax_y = fig_x.add_subplot(312)
    ax_y.plot(t_val, x_val[1, :])
    ax_theta = fig_x.add_subplot(313)
    ax_theta.plot(t_val, x_val[2, :])
    ax_x.set_ylabel("x (m)")
    ax_y.set_ylabel("y (m)")
    ax_theta.set_ylabel(r"$\theta$ (rad)")
    ax_theta.set_xlabel("t (s)")

    fig_u.show()
    fig_h.show()
    fig_x.show()
    return x_val, u_val, h_val, hdot_val


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
    parser.add_argument("--load_barrier_relu",
                        type=str,
                        default=None,
                        help="path to load barrier relu")
    parser.add_argument("--simulate", action="store_true")
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

    if args.load_barrier_relu is None:
        barrier_relu = utils.setup_relu((3, 8, 4, 1),
                                        params=None,
                                        negative_slope=0.1,
                                        bias=True)
        x_star = torch.tensor([0, 0, 0], dtype=dtype)
        c = 0.5
        # inf_norm_term = barrier.InfNormTerm.from_bounding_box(
        #     x_lo, x_up, 0.7)
        inf_norm_term = None
        epsilon = 0.2
    else:
        barrier_data = torch.load(args.load_barrier_relu)
        barrier_relu = utils.setup_relu(
            barrier_data["linear_layer_width"],
            params=None,
            negative_slope=barrier_data["negative_slope"],
            bias=barrier_data["bias"],
            dtype=dtype)
        barrier_relu.load_state_dict(barrier_data["state_dict"])
        x_star = barrier_data["x_star"]
        c = barrier_data["c"]
        epsilon = barrier_data["epsilon"]
        if barrier_data["inf_norm_term"] is not None:
            inf_norm_term = barrier.InfNormTerm(
                barrier_data["inf_norm_term"]["R"],
                barrier_data["inf_norm_term"]["p"])
        else:
            inf_norm_term = None
    barrier_system = control_barrier.ControlBarrier(dynamics_model,
                                                    barrier_relu)
    if args.simulate:
        x_val, u_val, h_val, hdot_val = simulate(barrier_system,
                                                 dt=0.01,
                                                 nT=5000,
                                                 x0=np.array([0.5, 0.5, 0]),
                                                 u0=np.array([0., 0]),
                                                 x_star=x_star,
                                                 c=c,
                                                 epsilon=epsilon,
                                                 inf_norm_term=inf_norm_term)
    else:
        verify_region_boundary = utils.box_boundary(x_lo, x_up)
        # unsafe region is a box region
        unsafe_region_cnstr = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        # unsafe_region_cnstr.Ain_input = torch.tensor(
        #     [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=dtype)
        # unsafe_region_cnstr.rhs_in = torch.tensor([0.6, 0.6, -0.4, -0.4],
        #                                           dtype=dtype)

        nominal_controller_nn = utils.setup_relu((3, 6, 8, 2),
                                                 params=None,
                                                 negative_slope=0.1,
                                                 bias=True,
                                                 dtype=dtype)
        nominal_control_state_samples = utils.uniform_sample_in_box(
            x_lo, x_up, 10000)
        u_star = torch.zeros((2, ), dtype=dtype)
        nominal_control_option = train_barrier.NominalControlOption(
            nominal_controller.NominalNNController(nominal_controller_nn,
                                                   x_star, u_star, u_lo, u_up),
            nominal_control_state_samples,
            weight=10.,
            margin=0.1,
            norm="max",
            nominal_control_loss_tol=0.8)

        dut = train_barrier.TrainBarrier(barrier_system, x_star, c,
                                         unsafe_region_cnstr,
                                         verify_region_boundary, epsilon,
                                         inf_norm_term, nominal_control_option)
        dut.enable_wandb = args.enable_wandb
        dut.max_iterations = args.max_iterations

        unsafe_state_samples = torch.zeros((0, 3), dtype=dtype)
        boundary_state_samples = torch.zeros((0, 3), dtype=dtype)
        deriv_state_samples = torch.zeros((0, 3), dtype=dtype)

        is_success = dut.train(unsafe_state_samples, boundary_state_samples,
                               deriv_state_samples)
    pass
