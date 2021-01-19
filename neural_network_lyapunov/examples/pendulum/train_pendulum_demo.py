import neural_network_lyapunov.examples.pendulum.pendulum as pendulum
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.train_lyapunov as train_lyapunov
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.train_utils as train_utils
import torch
import scipy.integrate
import numpy as np
import argparse
import os


def rotation_matrix(theta):
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    return torch.tensor([[c_theta, -s_theta], [s_theta, c_theta]],
                        dtype=torch.float64)


def generate_pendulum_dynamics_data(dt):
    """
    Generate the pairs (x[n], u[n]) -> (x[n+1])
    """
    dtype = torch.float64
    plant = pendulum.Pendulum(dtype)
    # We first generate the training data by simulating the system using some
    # LQR controller.
    Q = np.diag(np.array([1, 10.]))
    R = np.array([[1.]])
    lqr_gain = plant.lqr_control(Q, R)
    # Try many initial states. Some close to [pi, 0], some far away from
    # [pi, 0].
    x0s = [
        np.array([np.pi + 0.1, 0.2]),
        np.array([np.pi - 0.1, 0.3]),
        np.array([np.pi - 0.2, -0.3]),
        np.array([np.pi + 0.15, -0.3])
    ]
    x0s.append(np.array([np.pi - 0.5, 1.5]))
    x0s.append(np.array([np.pi + 0.5, 1.5]))
    x0s.append(np.array([np.pi + 1., 1.]))
    x_des = np.array([np.pi, 0])

    def converged(t, y):
        return np.linalg.norm(y - x_des) - 1E-2

    converged.terminal = True

    states = []
    controls = []
    next_states = []

    for x0 in x0s:
        result = scipy.integrate.solve_ivp(
            lambda t, x: plant.dynamics(x, lqr_gain @ (x - x_des)), (0, 10),
            x0,
            t_eval=np.arange(0, 10, dt),
            events=converged)
        states.append(torch.from_numpy(result.y[:, :-1]))
        controls.append(
            torch.from_numpy(lqr_gain @ (result.y[:, :-1] - x_des.reshape(
                (-1, 1)))))
        next_states.append(torch.from_numpy(result.y[:, 1:]))

    # Now take a grid of x and u, and compute the next state.
    theta_grid = np.linspace(-0.5 * np.pi, 2.5 * np.pi, 101)
    thetadot_grid = np.linspace(-5, 5, 101)
    u_grid = np.linspace(-15, 15, 401)

    for i in range(len(theta_grid)):
        for j in range(len(thetadot_grid)):
            for k in range(len(u_grid)):
                states.append(
                    torch.tensor([[theta_grid[i]], [thetadot_grid[j]]],
                                 dtype=dtype))
                controls.append(torch.tensor([[u_grid[k]]], dtype=dtype))
                result = scipy.integrate.solve_ivp(
                    lambda t, x: plant.dynamics(x, np.array([u_grid[k]])),
                    (0, dt), np.array([theta_grid[i], thetadot_grid[j]]))
                next_states.append(
                    torch.from_numpy(result.y[:, -1].reshape((-1, 1))))

    dataset_input = torch.cat(
        (torch.cat(states, dim=1), torch.cat(controls, dim=1)), dim=0).T
    dataset_output = torch.cat(next_states, dim=1).T
    return torch.utils.data.TensorDataset(dataset_input, dataset_output)


def train_forward_model(dynamics_model, model_dataset):
    state_equilibrium = torch.tensor([np.pi, 0], dtype=torch.float64)
    control_equilibrium = torch.tensor([0], dtype=torch.float64)
    # model_dataset contains the mapping from (x[n], u[n]) to x[n+1], but we
    # only need a mapping from (x[n], u[n]) to v[n+1]. So we regenerate a
    # dataset whose target only contains thetadot.
    (xu_inputs, x_next_outputs) = model_dataset[:]
    v_dataset = torch.utils.data.TensorDataset(
        xu_inputs, x_next_outputs[:, 1].reshape((-1, 1)))

    def compute_next_v(model, state_action):
        return model(state_action) - model(
            torch.cat((state_equilibrium, control_equilibrium)))

    utils.train_approximator(v_dataset,
                             dynamics_model,
                             compute_next_v,
                             batch_size=20,
                             num_epochs=100,
                             lr=0.001)


def generate_controller_dataset():
    """
    Generate the dataset to train a controller and value function.
    Simulate the pendulum dynamics (and cost-to-go) using an energy shaping +
    LQR controller. For each state on the simulated trajectory, obtain the
    control action and cost-to-go.
    """
    plant = pendulum.Pendulum(torch.float64)
    Q = 0.1 * np.diag([1, 10])
    R = 0.1 * np.array([[1]])
    x_des = np.array([np.pi, 0])

    def state_cost_dot(y, u):
        x = y[:2]
        x_des = np.array([np.pi, 0])
        # return [xdot; cost(x, u)]
        return np.hstack((plant.dynamics(x, u),
                          (x - x_des).dot(Q @ (x - x_des)) + u.dot(R @ u)))

    lqr_gain = plant.lqr_control(Q, R)

    def controller(x):
        if (x - x_des).dot(Q @ (x - x_des)) > 0.1:
            u = plant.energy_shaping_control(x, x_des, 1)
        else:
            u = lqr_gain @ (x - x_des)
        return u

    def converged(t, y):
        x = y[:2]
        return np.linalg.norm(x - x_des) - 1E-2

    converged.terminal = True
    # Now take a grid of initial states, simulate them using the controller.
    theta0 = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 11)
    thetadot0 = np.linspace(-2, 2, 11)
    states = []
    controls = []
    costs = []
    dt = 0.01
    for i in range(theta0.shape[0]):
        for j in range(thetadot0.shape[0]):
            result = scipy.integrate.solve_ivp(
                lambda t, y: state_cost_dot(y, controller(y[:2])), (0, 15),
                np.array([theta0[i], thetadot0[j], 0]),
                t_eval=np.arange(0, 15, dt),
                events=converged)
            if converged(0, result.y[:, -1]) <= 1E-4:
                states.append(result.y[:2, :])
                controls.append([
                    controller(result.y[:2, i])[0]
                    for i in range(result.y.shape[1])
                ])
                interval_cost = np.hstack(
                    (result.y[2, 1:] - result.y[2, :-1], 0))
                costs.append(np.cumsum(interval_cost[::-1])[::-1])
    return torch.from_numpy(np.hstack(states).T),\
        torch.from_numpy(np.hstack(controls).reshape((-1, 1))),\
        torch.from_numpy(np.hstack(costs).reshape((-1, 1)))


def train_controller_approximator(state_samples, control_samples,
                                  controller_relu, lr):
    """
    Given some state-action pairs, train a controller ϕ(x) − ϕ(x*) + u* to
    approximate these state-action pairs.
    """
    control_dataset = torch.utils.data.TensorDataset(state_samples,
                                                     control_samples)

    state_equilibrium = torch.tensor([np.pi, 0], dtype=torch.float64)
    control_equilibrium = torch.tensor([0], dtype=torch.float64)

    def compute_control(model, dataset):
        return model(dataset) - model(state_equilibrium) + control_equilibrium

    utils.train_approximator(control_dataset,
                             controller_relu,
                             compute_control,
                             batch_size=20,
                             num_epochs=400,
                             lr=lr)


def train_cost_approximator(state_samples, cost_samples, cost_relu, V_lambda):
    """
    Given many state-cost pairs, train a value approximator
    ϕ(x) − ϕ(x*)+λ|x − x*|₁
    """
    cost_dataset = torch.utils.data.TensorDataset(state_samples, cost_samples)
    state_equilibrium = torch.tensor([np.pi, 0], dtype=torch.float64)

    def compute_cost(model, data):
        return model(data) - model(state_equilibrium) + V_lambda * torch.norm(
            data - state_equilibrium, p=1, dim=1).reshape((-1, 1))

    utils.train_approximator(cost_dataset,
                             cost_relu,
                             compute_cost,
                             batch_size=20,
                             num_epochs=300,
                             lr=0.001)


def pendulum_closed_loop_dynamics(plant: pendulum.Pendulum, x: np.ndarray,
                                  controller_relu, x_equilibrium,
                                  u_equilibrium, u_lo, u_up):
    assert (isinstance(plant, pendulum.Pendulum))
    u_pre_saturation = controller_relu(torch.from_numpy(x)) -\
        controller_relu(x_equilibrium) + u_equilibrium
    u = torch.max(torch.min(u_pre_saturation, u_up), u_lo).detach().numpy()
    return plant.dynamics(x, u)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pendulum training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument("--load_dynamics_data",
                        type=str,
                        default=None,
                        help="path of the dynamics data")
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--generate_controller_cost_data", action="store_true")
    parser.add_argument("--train_controller_approximator", action="store_true")
    parser.add_argument("--train_cost_approximator", action="store_true")
    parser.add_argument("--load_controller_cost_data", action="store_true")
    parser.add_argument("--load_lyapunov_relu",
                        type=str,
                        default=None,
                        help="path of the saved lyapunov_relu state_dict()")
    parser.add_argument("--load_controller_relu",
                        type=str,
                        default=None,
                        help="path of the controller relu state_dict()")
    parser.add_argument("--pretrain_num_epochs",
                        type=int,
                        default=100,
                        help="number of epochs in pre-training on samples.")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5000,
        help="max number of iterations in searching for controller.")
    parser.add_argument("--search_R",
                        action="store_true",
                        help="search R when searching for controller.")
    parser.add_argument("--train_on_samples",
                        action="store_true",
                        help="pretrain Lyapunov controller on samples.")
    parser.add_argument("--enable_wandb", action="store_true")
    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dt = 0.01
    if args.generate_dynamics_data:
        model_dataset = generate_pendulum_dynamics_data(dt)
    if args.load_dynamics_data is not None:
        data = torch.load(args.load_dynamics_data)
        model_dataset = torch.utils.data.TensorDataset(data["input"],
                                                       data["output"])
    # Setup forward dynamics model
    dynamics_model = utils.setup_relu((3, 5, 5, 1),
                                      params=None,
                                      negative_slope=0.01,
                                      bias=True,
                                      dtype=torch.float64)
    if args.train_forward_model:
        train_forward_model(dynamics_model, model_dataset)
    else:
        dynamics_model_data = torch.load(
            dir_path + "/data/pendulum_second_order_forward_relu2.pt")
        dynamics_model = utils.setup_relu(
            dynamics_model_data["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["negative_slope"],
            bias=True,
            dtype=torch.float64)
        dynamics_model.load_state_dict(dynamics_model_data["state_dict"])
    if args.generate_controller_cost_data:
        state_samples, control_samples, cost_samples =\
            generate_controller_dataset()
    elif args.load_controller_cost_data:
        controller_cost_data = torch.load(
            dir_path + "/data/pendulum_controller_cost_data.pt")
        state_samples = controller_cost_data["state_samples"]
        control_samples = controller_cost_data["control_samples"]
        cost_samples = controller_cost_data["cost_samples"]

    V_lambda = 0.8
    controller_relu = utils.setup_relu((2, 3, 2, 1),
                                       params=None,
                                       negative_slope=0.1,
                                       bias=True,
                                       dtype=torch.float64)
    if args.train_controller_approximator:
        train_controller_approximator(state_samples,
                                      control_samples,
                                      controller_relu,
                                      lr=0.001)
    elif args.load_controller_relu is not None:
        controller_data = torch.load(args.load_controller_relu)
        controller_relu = utils.setup_relu(
            controller_data["linear_layer_width"],
            params=None,
            negative_slope=controller_data["negative_slope"],
            bias=True,
            dtype=torch.float64)
        controller_relu.load_state_dict(controller_data["state_dict"])

    plant = pendulum.Pendulum(torch.float64)
    lqr_gain = plant.lqr_control(np.diag([1., 10.]), np.array([[1.]]))

    lyapunov_relu = utils.setup_relu((2, 8, 8, 6, 1),
                                     params=None,
                                     negative_slope=0.1,
                                     bias=True,
                                     dtype=torch.float64)
    R = torch.cat((rotation_matrix(np.pi / 4), rotation_matrix(np.pi / 10)),
                  dim=0)
    if args.train_cost_approximator:
        train_cost_approximator(state_samples, cost_samples, lyapunov_relu,
                                V_lambda)
    elif args.load_lyapunov_relu is not None:
        lyapunov_data = torch.load(args.load_lyapunov_relu)
        lyapunov_relu = utils.setup_relu(
            lyapunov_data["linear_layer_width"],
            params=None,
            negative_slope=lyapunov_data["negative_slope"],
            bias=True,
            dtype=torch.float64)
        lyapunov_relu.load_state_dict(lyapunov_data["state_dict"])
        V_lambda = lyapunov_data["V_lambda"]
        R = lyapunov_data["R"]

    # Now train the controller and Lyapunov function together
    q_equilibrium = torch.tensor([np.pi], dtype=torch.float64)
    u_equilibrium = torch.tensor([0], dtype=torch.float64)
    x_lo = torch.tensor([np.pi - 0.7 * np.pi, -3.], dtype=torch.float64)
    x_up = torch.tensor([np.pi + 0.7 * np.pi, 3.], dtype=torch.float64)
    u_lo = torch.tensor([-20], dtype=torch.float64)
    u_up = torch.tensor([20], dtype=torch.float64)
    forward_system = relu_system.ReLUSecondOrderSystemGivenEquilibrium(
        torch.float64, x_lo, x_up, u_lo, u_up, dynamics_model, q_equilibrium,
        u_equilibrium, dt)
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy())
    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
        closed_loop_system, lyapunov_relu)

    if args.search_R:
        R_options = train_lyapunov.SearchROptions(R.shape, epsilon=0.01)
        R_options.set_variable_value(R.detach().numpy())
    else:
        R_options = train_lyapunov.FixedROptions(R)

    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, controller_relu,
                                        x_lo, x_up, u_lo, u_up)
    dut = train_lyapunov.TrainLyapunovReLU(lyapunov_hybrid_system, V_lambda,
                                           closed_loop_system.x_equilibrium,
                                           R_options)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-5
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.5
    dut.lyapunov_derivative_epsilon = 0.001
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    state_samples_all = utils.get_meshgrid_samples(x_lo,
                                                   x_up, (51, 51),
                                                   dtype=torch.float64)
    dut.output_flag = True
    if args.train_on_samples:
        dut.train_lyapunov_on_samples(state_samples_all,
                                      num_epochs=args.pretrain_num_epochs,
                                      batch_size=50)

    dut.enable_wandb = args.enable_wandb
    dut.add_derivative_adversarial_state = True
    dut.lyapunov_derivative_mip_cost_weight = 0.
    dut.add_positivity_adversarial_state = True
    dut.lyapunov_positivity_mip_cost_weight = 0.
    dut.train(torch.empty((0, 2), dtype=torch.float64))
    pass
