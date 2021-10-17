import neural_network_lyapunov.examples.pendulum.pendulum as pendulum
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.control_lyapunov as control_lyapunov
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import torch
import numpy as np
import argparse
import scipy.integrate
import gurobipy


class ControlAffinePendulum(
        control_affine_system.SecondOrderControlAffineSystem):
    """
    The dynamics is v̇ = ϕ_a(x) − ϕ_a(x*) + b * (u − u*)
    """
    def __init__(self, x_lo, x_up, u_lo, u_up, phi_a, b_val: torch.Tensor,
                 method: mip_utils.PropagateBoundsMethod):
        super(ControlAffinePendulum, self).__init__(x_lo, x_up, u_lo, u_up)
        self.x_equilibrium = torch.tensor([np.pi, 0], dtype=self.dtype)
        self.u_equilibrium = torch.tensor([0], dtype=self.dtype)
        self.phi_a = phi_a
        assert (isinstance(b_val, torch.Tensor))
        assert (b_val.shape == (1, 1))
        self.b_val = b_val
        self.method = method
        self.relu_free_pattern_a = relu_to_optimization.ReLUFreePattern(
            self.phi_a, self.dtype)

    def a(self, x):
        return self.phi_a(x) - self.phi_a(
            self.x_equilibrium) - self.b_val @ self.u_equilibrium

    def b(self, x):
        return self.b_val

    def _mixed_integer_constraints_v(self):
        mip_cnstr_a = self.relu_free_pattern_a.output_constraint(
            self.x_lo, self.x_up, self.method)
        a_delta = -self.phi_a(self.x_equilibrium)
        mip_cnstr_a.Cout += a_delta
        mip_cnstr_b_flat = gurobi_torch_mip.MixedIntegerConstraintsReturn()
        mip_cnstr_b_flat.Cout = self.b_val.reshape((-1, ))
        a_lo = mip_cnstr_a.nn_output_lo + a_delta
        a_up = mip_cnstr_a.nn_output_up + a_delta
        b_lo = self.b_val.reshape((-1, ))
        b_up = self.b_val.reshape((-1, ))
        return mip_cnstr_a, mip_cnstr_b_flat, a_lo, a_up, b_lo, b_up


def rotation_matrix(theta):
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    return torch.tensor([[c_theta, -s_theta], [s_theta, c_theta]],
                        dtype=torch.float64)


def generate_pendulum_dynamics_data():
    dtype = torch.float64
    plant = pendulum.Pendulum(dtype)

    xu_samples = utils.uniform_sample_in_box(
        torch.tensor([-0.5 * np.pi, -5, -20], dtype=dtype),
        torch.tensor([2.5 * np.pi, 5, 20], dtype=dtype), 100000)
    xdot = torch.vstack([
        plant.dynamics(xu_samples[i, :2], xu_samples[i, 2:])
        for i in range(xu_samples.shape[0])
    ])
    return torch.utils.data.TensorDataset(xu_samples, xdot)


def train_forward_model(dynamics_model: ControlAffinePendulum, model_dataset):
    (xu_inputs, x_next_outputs) = model_dataset[:]
    v_dataset = torch.utils.data.TensorDataset(
        xu_inputs, x_next_outputs[:, 1].reshape((-1, 1)))

    def compute_vdot(phi_a, state_action, b_val):
        x, u = torch.split(state_action, [2, 1], dim=1)
        vdot = phi_a(x) - phi_a(dynamics_model.x_equilibrium) + b_val * u
        return vdot

    utils.train_approximator(v_dataset,
                             dynamics_model.phi_a,
                             compute_vdot,
                             batch_size=30,
                             num_epochs=100,
                             lr=0.001,
                             additional_variable=list(dynamics_model.b_val),
                             output_fun_args=dict(b_val=dynamics_model.b_val))


def simulate_pendulum(x0, T, clf, V_lambda, R, epsilon):
    dtype = torch.float64
    plant = pendulum.Pendulum(dtype)
    x_equilibrium = torch.tensor([np.pi, 0], dtype=dtype)

    def dyn(t, x):
        x_torch = torch.from_numpy(x)
        # Now formulate a QP
        # min uᵀu
        # s.t Vdot(x, u) <= 0
        prog = gurobipy.Model()
        u = prog.addVar(lb=clf.system.u_lo[0].item(),
                        ub=clf.system.u_up[0].item())
        dphi_dx = utils.relu_network_gradient(clf.lyapunov_relu, x_torch)
        dl1_dx = V_lambda * utils.l1_gradient(
            R @ (x_torch - x_equilibrium)) @ R
        f = clf.system.f(x_torch)
        G = clf.system.G(x_torch)
        V = clf.lyapunov_value(x_torch, x_equilibrium, V_lambda, R=R)
        for i in range(dphi_dx.shape[0]):
            for j in range(dl1_dx.shape[0]):
                dVdx = dphi_dx[i][0] + dl1_dx[j]
                prog.addLConstr(gurobipy.LinExpr((dVdx @ G).tolist(), [u]),
                                sense=gurobipy.GRB.LESS_EQUAL,
                                rhs=-(dVdx @ f).item() - epsilon * 0.9 * V)
        prog.setObjective(u * u, sense=gurobipy.GRB.MINIMIZE)
        prog.setParam(gurobipy.GRB.Param.OutputFlag, False)
        prog.optimize()
        assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
        u_val = np.array([u.x])
        xdot = plant.dynamics(x, u_val)
        return xdot

    def reach_goal(t, x):
        return np.linalg.norm(x - np.array([np.pi, 0])) - 1E-3

    reach_goal.terminal = True

    return scipy.integrate.solve_ivp(dyn, [0, T],
                                     x0,
                                     t_eval=np.arange(0, T, 0.001),
                                     events=reach_goal,
                                     max_step=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pendulum clf training demo")
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

    if args.generate_dynamics_data:
        model_dataset = generate_pendulum_dynamics_data()
    if args.load_dynamics_data is not None:
        model_dataset = torch.load(args.load_dynamics_data)

    phi_a = utils.setup_relu((2, 6, 6, 1),
                             params=None,
                             negative_slope=0.01,
                             bias=True,
                             dtype=dtype)
    b_val = torch.tensor([[1]], dtype=dtype)
    x_lo = torch.tensor([np.pi - 1.1 * np.pi, -5], dtype=dtype)
    x_up = torch.tensor([np.pi + 1.1 * np.pi, 5], dtype=dtype)
    u_lo = torch.tensor([-20], dtype=dtype)
    u_up = torch.tensor([20], dtype=dtype)
    x_equilibrium = torch.tensor([np.pi, 0], dtype=dtype)
    u_equilibrium = torch.tensor([0], dtype=dtype)
    dynamics_model = ControlAffinePendulum(
        x_lo,
        x_up,
        u_lo,
        u_up,
        phi_a,
        b_val,
        method=mip_utils.PropagateBoundsMethod.IA)

    if args.train_forward_model is not None:
        dynamics_model.b_val.requires_grad = True
        train_forward_model(dynamics_model, model_dataset)
        linear_layer_width_a, negative_slope_a, bias_a = \
            utils.extract_relu_structure(phi_a)
        torch.save(
            {
                "phi_a": {
                    "linear_layer_width": linear_layer_width_a,
                    "state_dict": phi_a.state_dict(),
                    "negative_slope": negative_slope_a,
                    "bias": bias_a
                },
                "b_val": dynamics_model.b_val
            }, args.train_forward_model)
        dynamics_model.b_val.requires_grad = False

    elif args.load_forward_model is not None:
        dynamics_model_data = torch.load(args.load_forward_model)
        dynamics_model.phi_a = utils.setup_relu(
            dynamics_model_data["phi_a"]["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["phi_a"]["negative_slope"],
            bias=dynamics_model_data["phi_a"]["bias"],
            dtype=dtype)
        dynamics_model.phi_a.load_state_dict(
            dynamics_model_data["phi_a"]["state_dict"])
        dynamics_model.b_val = dynamics_model_data["b_val"]

    if args.load_lyapunov_relu is None:
        V_lambda = 0.5
        lyapunov_relu = utils.setup_relu((2, 8, 8, 1),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=dtype)
        R = torch.cat(
            (rotation_matrix(np.pi / 4), rotation_matrix(np.pi / 10)), dim=0)
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

    R_options = r_options.FixedROptions(R)

    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, None, x_lo, x_up,
                                        u_lo, u_up)

    dut = train_lyapunov.TrainLyapunovReLU(lyapunov_hybrid_system, V_lambda,
                                           x_equilibrium, R_options)

    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_epsilon = 0.1
    dut.lyapunov_derivative_mip_clamp_min = -2.
    dut.output_flag = True
    dut.max_iterations = args.max_iterations
    dut.enable_wandb = args.enable_wandb
    state_samples_all = torch.empty((0, 2), dtype=dtype)
    dut.train(state_samples_all)

    pass
