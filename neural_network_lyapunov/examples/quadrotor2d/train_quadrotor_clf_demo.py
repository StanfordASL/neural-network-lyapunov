import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.examples.quadrotor2d.quadrotor_2d as \
    quadrotor_2d
import neural_network_lyapunov.examples.quadrotor2d.control_affine_quadrotor\
    as control_affine_quadrotor
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.control_lyapunov as control_lyapunov
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.integrator as integrator

import torch
import numpy as np
import argparse
import gurobipy


def simulate(lyapunov_hybrid_system, dt, nT, x0, x_equilibrium, V_lambda,
             deriv_epsilon, R):
    def compute_control(x, u_des):
        x_torch = torch.from_numpy(x)
        prog = gurobipy.Model()
        u = prog.addVars(2,
                         lb=lyapunov_hybrid_system.system.u_lo.tolist(),
                         ub=lyapunov_hybrid_system.system.u_up.tolist())
        f = lyapunov_hybrid_system.system.f(x_torch).detach().numpy()
        G = lyapunov_hybrid_system.system.G(x_torch).detach().numpy()
        dhdx = lyapunov_hybrid_system._lyapunov_gradient(
            x_torch, x_equilibrium, V_lambda, R,
            zero_tol=1E-6).detach().numpy()
        V = lyapunov_hybrid_system.lyapunov_value(x_torch,
                                                  x_equilibrium,
                                                  V_lambda,
                                                  R=R).detach().numpy()
        prog.addMConstrs(dhdx @ G, [u[0], u[1]],
                         sense=gurobipy.GRB.LESS_EQUAL,
                         b=(-deriv_epsilon * V - dhdx @ f))
        prog.setParam(gurobipy.GRB.Param.OutputFlag, False)
        prog.setObjective((u[0] - u_des[0]) * (u[0] - u_des[0]) +
                          (u[1] - u_des[1]) * (u[1] - u_des[1]))
        prog.optimize()
        assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
        u_val = np.array([u[0].x, u[1].x])
        return u_val

    x_val = np.zeros((6, nT))
    x_val[:, 0] = x0
    u_val = np.zeros((2, nT))
    for i in range(nT - 1):
        if i == 0:
            u_des = np.array([0, 0.])
        else:
            u_des = u_val[:, i - 1]
        x_val[:, i + 1], u_val[:, i] = integrator.rk4_constant_control(
            lambda x, u: lyapunov_hybrid_system.system.dynamics(
                torch.from_numpy(x), torch.from_numpy(u)).detach().numpy(),
            lambda x: compute_control(x, u_des),
            x_val[:, i],
            dt,
            constant_control_steps=1)
    u_val[:, -1] = compute_control(x_val[:, -1], u_val[:, -2])
    V_val = lyapunov_hybrid_system.lyapunov_value(torch.from_numpy(x_val.T),
                                                  x_equilibrium,
                                                  V_lambda,
                                                  R=R)
    return x_val, u_val, V_val


def generate_quadrotor_dynamics_data():
    dtype = torch.float64
    plant = quadrotor_2d.Quadrotor2D(dtype)

    xu_samples = utils.uniform_sample_in_box(
        torch.tensor([0, 0, -1.2 * np.pi, 0, 0, -5, 0, 0], dtype=dtype),
        torch.tensor([
            0, 0, 1.2 * np.pi, 0, 0, 5, plant.mass * plant.gravity / 2 * 3,
            plant.mass * plant.gravity / 2 * 3
        ],
                     dtype=dtype), 100000)

    xdot = torch.vstack([
        plant.dynamics(xu_samples[i, :6], xu_samples[i, 6:])
        for i in range(xu_samples.shape[0])
    ])
    return torch.utils.data.TensorDataset(xu_samples, xdot)


def train_forward_model(
        dynamics_model: control_affine_quadrotor.ControlAffineQuadrotor2d,
        model_dataset):
    (xu_inputs, xdot_outputs) = model_dataset[:]

    v_dataset = torch.utils.data.TensorDataset(xu_inputs, xdot_outputs[:, 3:])

    def compute_vdot(phi_b, state_action):
        x, u = torch.split(state_action, [6, 2], dim=1)
        a_val = -phi_b(torch.tensor([0], dtype=torch.float64)).reshape(
            (3, 2)) @ dynamics_model.u_equilibrium
        vdot = (phi_b(x[:, 2].unsqueeze(1)).reshape(
            (-1, 3, 2)) @ u.unsqueeze(2)).squeeze(2) + a_val
        return vdot

    utils.train_approximator(v_dataset,
                             dynamics_model.phi_b,
                             compute_vdot,
                             batch_size=30,
                             num_epochs=200,
                             lr=0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="quadrotor2d clf training demo")
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
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    dtype = torch.float64

    if args.generate_dynamics_data:
        model_dataset = generate_quadrotor_dynamics_data()
    if args.load_dynamics_data is not None:
        model_dataset = torch.load(args.load_dynamics_data)

    phi_b = utils.setup_relu((1, 15, 10, 6),
                             params=None,
                             negative_slope=0.01,
                             bias=True,
                             dtype=dtype)
    plant = quadrotor_2d.Quadrotor2D(dtype)
    a_val = torch.tensor([0, -plant.gravity, 0], dtype=dtype)
    x_lo = torch.tensor([-0.7, -0.7, -np.pi * 0.5, -6, -6, -3], dtype=dtype)
    x_up = -x_lo
    u_lo = torch.tensor([0, 0], dtype=dtype)
    u_up = torch.tensor([1, 1], dtype=dtype) * plant.mass * plant.gravity * 1.5
    x_equilibrium = torch.zeros(6, dtype=dtype)
    u_equilibrium = torch.tensor(
        [1, 1], dtype=dtype) * plant.mass * plant.gravity * 0.5

    if args.train_forward_model is not None:
        dynamics_model = control_affine_quadrotor.ControlAffineQuadrotor2d(
            x_lo,
            x_up,
            u_lo,
            u_up,
            phi_b,
            u_equilibrium,
            method=mip_utils.PropagateBoundsMethod.IA)
        train_forward_model(dynamics_model, model_dataset)
        linear_layer_width_b, negative_slope_b, bias_b = \
            utils.extract_relu_structure(phi_b)
        torch.save(
            {
                "phi_b": {
                    "linear_layer_width": linear_layer_width_b,
                    "state_dict": phi_b.state_dict(),
                    "negative_slope": negative_slope_b,
                    "bias": bias_b
                },
            }, args.train_forward_model)

    elif args.load_forward_model is not None:
        dynamics_model_data = torch.load(args.load_forward_model)
        phi_b = utils.setup_relu(
            dynamics_model_data["phi_b"]["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["phi_b"]["negative_slope"],
            bias=dynamics_model_data["phi_b"]["bias"],
            dtype=dtype)
        phi_b.load_state_dict(dynamics_model_data["phi_b"]["state_dict"])
        dynamics_model = control_affine_quadrotor.ControlAffineQuadrotor2d(
            x_lo,
            x_up,
            u_lo,
            u_up,
            phi_b,
            u_equilibrium,
            method=mip_utils.PropagateBoundsMethod.IA)

    if args.load_lyapunov_relu is None:
        V_lambda = 0.5
        lyapunov_relu = utils.setup_relu((6, 12, 12, 1),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=dtype)
        _, S = plant.lqr_control(np.diag([1, 1, 1, 10, 10, 10.]),
                                 np.diag([10., 10.]), x_equilibrium,
                                 u_equilibrium)
        R = torch.vstack(
            (torch.from_numpy(S),
             torch.tensor([[1, 1, 0, 1, 1, 0], [0, 0, 1, 0, 0, 10]],
                          dtype=dtype)))
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
        control_lyapunov.SubgradientPolicy(np.array([-0.5, 0., 0.5])))

    R_options = r_options.FixedROptions(R)

    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, None, x_lo, x_up,
                                        u_lo, u_up)

    if args.simulate:
        x_val, u_val, V_val = simulate(lyapunov_hybrid_system,
                                       dt=0.001,
                                       nT=2000,
                                       x0=np.array([0.5, 0.5, 0.2, 0, 0., 0.]),
                                       x_equilibrium=x_equilibrium,
                                       V_lambda=V_lambda,
                                       deriv_epsilon=0.1,
                                       R=R)
    else:
        dut = train_lyapunov.TrainLyapunovReLU(lyapunov_hybrid_system,
                                               V_lambda, x_equilibrium,
                                               R_options)

        dut.lyapunov_positivity_mip_pool_solutions = 1
        dut.lyapunov_derivative_mip_pool_solutions = 1
        dut.lyapunov_derivative_epsilon = 0.1
        dut.lyapunov_derivative_mip_clamp_min = -1.
        dut.lyapunov_derivative_mip_cost_weight = 1.
        dut.output_flag = True
        dut.learning_rate = 0.003
        dut.max_iterations = args.max_iterations
        dut.enable_wandb = args.enable_wandb
        state_samples_all = torch.empty((0, 6), dtype=dtype)
        dut.train(state_samples_all)

    pass
