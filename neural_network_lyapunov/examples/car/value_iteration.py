import numpy as np
import neural_network_lyapunov.examples.car.unicycle_traj_opt as\
    unicycle_traj_opt
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.examples.car.train_unicycle_demo as\
    train_unicycle_demo
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.examples.car.unicycle as unicycle
import pydrake.solvers.mathematicalprogram as mp
import math
import argparse
import torch
import matplotlib.pyplot as plt


def value_iteration(nx, r, u_optimal, epsilon=0.0001, discount_factor=0.95):
    """
    Value Iteration Algorithm.
    Args:
        nx: number of state in the environment.
        r: numpy array of size (nx, nx). immediate cost from state x to x'.
        u_optimal: numpy array of size (nx, nx, u_dim). u[i, j] is the
        optimal control action from state s[i] to s[j].
        epsilon: threshold for convergence.
        discount_factor: Gamma discount factor.
    Returns:
        policy: numpy array of size (nx, u_dim). policy[i] is the optimal
        control of state s[i].
        V: numpy array of size nx. V[i] is the optimal value function of
        state s[i].
    """

    def one_step_lookahead(V, r, i):
        j = np.argmin(r[i, :] + discount_factor * V)
        v = r[i, j] + discount_factor * V[j]
        return j, v

    # start with inital value function and intial policy

    V = np.zeros(nx)
    policy = np.zeros((nx, 2))

    n = 0
    # while not the optimal policy
    while True:
        print('Iteration: ', n)
        # for stopping condition
        delta = 0

        # loop over state space
        for i in range(nx):

            j, v = one_step_lookahead(V, r, i)

            # get the biggest difference between best action value and our old
            # value function
            if v != V[i]:
                delta = max(delta, abs(v - V[i]))

            # apply bellman optimality eqn
            V[i] = v
            policy[i, :] = u_optimal[i, j, :]

        # if optimal value function
        if (delta < epsilon):
            break
        n += 1

    return V, policy


def greedy_policy(s, u, network, V_lambda, R, dt=0.1):
    n_state = 5000
    n_action = 100
    dp_states = s[:, np.random.randint(s.shape[1], size=n_state)]
    dp_actions = np.zeros((2, n_state))
    plant = unicycle.Unicycle(torch.float64)
    for i in range(n_state):
        if i % 100 == 0:
            print("DP state ", i)
        si = dp_states[:,i]
        actions = u[:,np.random.randint(u.shape[1], size=n_action)]
        V = np.zeros(n_action)
        for j in range(n_action):
            s_next = plant.next_pose(si, actions[:, j], dt)
            r = unicycle_traj_opt.cost_trajectory(1, actions[:, j])
            with torch.no_grad():
                V_next = network(torch.from_numpy(s_next)) - network(
                    torch.zeros((3,), dtype=torch.float64)) + V_lambda * torch.norm(R @ torch.from_numpy(s_next).T, p=1, dim=0).reshape(
                        (-1, 1))
            V[j] = r[0] + V_next.detach().numpy()[0][0]
        ind = np.argmin(V)
        dp_actions[:, i] = actions[:, ind]
    return dp_states, dp_actions



def calculate_r_complete(
    ns,
    n0,
    n1,
    n2,
    prog,
    initial_val_constraint,
    final_val_constraint,
    x,
    u,
        dt):
    """
    @return:
    r: numpy array of size (nx, nx). immediate cost from state x to x'.
    u_optimal: numpy array of size (nx, nx, u_dim). u[i, j] is the
    optimal control action from state s[i] to s[j].
    """

    # r = np.ones((ns, ns))*1000
    r = np.full((ns, ns), np.inf)
    u_optimal = np.full((ns, ns, 2), np.nan)
    for i in range(ns):
        i0, i1, i2 = flat_to_3d(i)
        if i % 100 == 0:
            print("r iteration: ", i)
        for j in range(ns):
            j0, j1, j2 = flat_to_3d(j)
            if i != j and (i0 - n0) <= j0 <= (i0 + n0) and \
                    (i1 - n1) <= j1 <= (i1 + n1) and \
                    (i2 - n2) <= j2 <= (i2 + n2):
                initial_val_constraint.evaluator(
                ).set_bounds(s[:, i], s[:, i])
                final_val_constraint.evaluator(
                ).set_bounds(s[:, j], s[:, j])
                prog.SetInitialGuess(
                    x, np.linspace(s[:, i], s[:, j], nT).T)
                prog.SetInitialGuess(dt, dt_max * np.ones((nT - 1,)))
                result = mp.Solve(prog)
                if result.is_success():
                    r[i, j] = result.get_optimal_cost()
                    u_optimal[i, j, :] = result.GetSolution(u)[:, 0]
            #     else:
            #         r[i, j] = np.inf
            # else:
            #     r[i, j] = np.inf
    return r, u_optimal


def calculate_r_duplicate(
    ns,
    n0,
    n1,
    n2,
    prog,
    initial_val_constraint,
    final_val_constraint,
    x,
    u,
        dt):
    """
    @return:
    r: numpy array of size (nx, nx). immediate cost from state x to x'.
    u_optimal: numpy array of size (nx, nx, u_dim). u[i, j] is the
    optimal control action from state s[i] to s[j].
    """

    r = np.full((ns, ns), np.inf)
    u_optimal = np.full((ns, ns, 2), np.nan)
    # Calculate a quarter of a duplicated block at the four corners
    i01s = [[0, 0], [0, n_grid_xy - 1],
            [n_grid_xy - 1, 0], [n_grid_xy - 1, n_grid_xy - 1]]
    for i2 in range(n_grid_angle):
        print("i2 iteration: ", i2)
        # Construct duplicated 3D block
        r3d = np.full((2 * n0 + 1, 2 * n1 + 1, 2 * n2 + 1), np.inf)
        u3d = np.full((2 * n0 + 1, 2 * n1 + 1, 2 * n2 + 1, 2), np.nan)
        for i01 in i01s:
            i0 = i01[0]
            i1 = i01[1]
            i = threeD_to_flat(i0, i1, i2)

            for j in range(ns):
                j0, j1, j2 = flat_to_3d(j)

                if i != j and (i0 - n0) <= j0 <= (i0 + n0) and \
                        (i1 - n1) <= j1 <= (i1 + n1) and \
                        (i2 - n2) <= j2 <= (i2 + n2):
                    initial_val_constraint.evaluator(
                    ).set_bounds(s[:, i], s[:, i])
                    final_val_constraint.evaluator(
                    ).set_bounds(s[:, j], s[:, j])
                    prog.SetInitialGuess(
                        x, np.linspace(s[:, i], s[:, j], nT).T)
                    prog.SetInitialGuess(dt, dt_max * np.ones((nT - 1,)))
                    result = mp.Solve(prog)

                    if result.is_success():
                        optimal_cost = result.get_optimal_cost()
                        r3d[j0 - i0 + n0, j1 - i1 + n1,
                            j2 - i2 + n2] = optimal_cost
                        r[i, j] = optimal_cost

                        optimal_control = result.GetSolution(u)[:, 0]
                        u3d[j0 - i0 + n0, j1 - i1 + n1,
                            j2 - i2 + n2] = optimal_control
                        u_optimal[i, j] = optimal_control

        for i0 in range(n_grid_xy):
            for i1 in range(n_grid_xy):
                if [i0, i1] not in i01:
                    i = threeD_to_flat(i0, i1, i2)
                    for j in range(ns):
                        j0, j1, j2 = flat_to_3d(j)
                        if i != j and 0 <= j0 - i0 + \
                                n0 < r3d.shape[0] and \
                                0 <= j1 - i1 + n1 < r3d.shape[1] and \
                                0 <= j2 - i2 + n2 < r3d.shape[2]:
                            r[i, j] = r3d[j0 - i0 + n0, j1 - i1 + n1,
                                          j2 - i2 + n2]
                            u_optimal[i, j] = u3d[j0 - i0 +
                                                  n0, j1 - i1 + n1,
                                                  j2 - i2 + n2]

    return r, u_optimal


def flat_to_3d(i):
    i0 = i // (n_grid_angle * n_grid_xy)
    i1 = (i - n_grid_angle * n_grid_xy * i0) // n_grid_angle
    i2 = i - n_grid_angle * n_grid_xy * i0 - n_grid_angle * i1
    return i0, i1, i2


def threeD_to_flat(i0, i1, i2):
    return n_grid_angle * n_grid_xy * i0 + n_grid_angle * i1 + i2


def plot_V(V, n_angle=20):
    V_copy = V.reshape((n_grid_xy, n_grid_xy, n_grid_angle))
    x, y = np.meshgrid(np.linspace(x_lo[0], x_up[0], n_grid_xy),
                       np.linspace(x_lo[1], x_up[1], n_grid_xy))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(x, y, V_copy[:, :, n_angle],
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('V')
    ax.set_title('Surface plot')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(x, y, V_copy[:, :, n_angle])
    fig.colorbar(cp)
    ax.set_title('Value Function Contours')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def plot_u(u, n_angle=20):
    u_copy = u.reshape((n_grid_xy, n_grid_xy, n_grid_angle, 2))
    theta = np.linspace(x_lo[2], x_up[2], n_grid_angle)[n_angle]

    x, y = np.meshgrid(np.linspace(x_lo[0], x_up[0], n_grid_xy),
                       np.linspace(x_lo[1], x_up[1], n_grid_xy))

    plt.quiver(x, y, np.cos(u_copy[:, :, n_angle, 1]), np.sin(
        u_copy[:, :, n_angle, 1]))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Vector Field of Yaw Rate with Heading " +
              str(theta * 180 / np.pi))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_r",
        help="load precomputed r",
        action="store_true")
    parser.add_argument(
        "--load_V",
        help="load computed V",
        action="store_true")
    parser.add_argument(
        "--r_complete",
        help="calculate r using complete without duplicate",
        action="store_true")
    parser.add_argument(
        "--train_approximator",
        help="train function approximator for controller and cost",
        action="store_true")
    parser.add_argument(
        "--nT", type=int,
        help="number of time step for trajectory optimization",
        default=5)
    parser.add_argument(
        "--n_grid_xy", type=int,
        help="number of grids along x/y axis",
        default=20)
    parser.add_argument(
        "--n_grid_angle", type=int,
        help="number of grids along angle axis",
        default=60)
    args = parser.parse_args()

    epsilon = 1e-4

    nT = args.nT
    u_lo = np.array([0, -0.15 * np.pi])
    u_up = np.array([1, 0.15 * np.pi])
    x_lo = np.array([-1, -1, -1.05 * np.pi])
    x_up = np.array([1, 1, 1.05 * np.pi])
    dt_min = 0.001
    dt_max = 0.1
    prog, initial_val_constraint, final_val_constraint, x, u, dt =\
        unicycle_traj_opt.construct_traj_opt(
            nT, u_lo, u_up, dt_min, dt_max)

    n_grid_xy = args.n_grid_xy
    n_grid_angle = args.n_grid_angle

    print("n_grid_xy: {}, n_grid_angle: {}". format(n_grid_xy, n_grid_angle))

    s = np.array(np.meshgrid(np.linspace(x_lo[0], x_up[0], n_grid_xy),
                             np.linspace(x_lo[1], x_up[1], n_grid_xy),
                             np.linspace(x_lo[2], x_up[2], n_grid_angle)))
    # ss = s.copy()
    a = np.array(np.meshgrid(np.linspace(u_lo[0], u_up[0], n_grid_xy),
                             np.linspace(u_lo[1], u_up[1], n_grid_angle)))
    a = a.reshape((2, n_grid_xy * n_grid_angle))
    s = s.reshape((3, n_grid_xy * n_grid_xy * n_grid_angle))
    ns = s.shape[1]
    n0 = math.ceil(u_up[0] * dt_max * (nT - 1) /
                   (x_up[0] - x_lo[0]) * (n_grid_xy - 1))
    n1 = math.ceil(u_up[0] * dt_max * (nT - 1) /
                   (x_up[1] - x_lo[1]) * (n_grid_xy - 1))
    n2 = math.ceil(u_up[1] * dt_max * (nT - 1) /
                   (x_up[2] - x_lo[2]) * (n_grid_angle - 1))

    if args.r_complete:
        folder_name = "neural_network_lyapunov/examples/car/value/complete/"
    else:
        folder_name = "neural_network_lyapunov/examples/car/value/"

    if args.load_V:
        V = np.load(
            folder_name + "V_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max) +
            ".npy")

        plot_V(V)

        s = np.load(
            folder_name + "s_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max) +
            ".npy")
        # policy = np.load(
        #     folder_name + "policy_" +
        #     str(nT) +
        #     "_" +
        #     str(n_grid_xy) +
        #     "_" +
        #     str(n_grid_angle) +
        #     "_" +
        #     str(dt_max) +
        #     ".npy")
        # plot_u(policy, 18)
    else:
        if args.load_r:
            # r = np.random.random((ns, ns))
            # u_optimal = np.random.random((ns, ns, 2))
            r = np.load(
                folder_name + "r_" +
                str(nT) +
                "_" +
                str(n_grid_xy) +
                "_" +
                str(n_grid_angle) +
                "_" +
                str(dt_max) +
                ".npy")

            # rc = np.load(
            #     folder_name + "complete/r_" +
            #     str(nT) +
            #     "_" +
            #     str(n_grid_xy) +
            #     "_" +
            #     str(n_grid_angle) +
            #     "_" +
            #     str(dt_max) +
            #     ".npy")
            # r100 = r[:100, :]
            # Check duplicate code matches complete code
            # print((np.absolute(r100[np.isfinite(r100)] -
            #                    rc[np.isfinite(rc)]) < epsilon).all())

            u_optimal = np.load(
                folder_name + "u_" +
                str(nT) +
                "_" +
                str(n_grid_xy) +
                "_" +
                str(n_grid_angle) +
                "_" +
                str(dt_max) +
                ".npy")
        else:
            if args.r_complete:
                r, u_optimal = calculate_r_complete(
                    ns, n0, n1, n2, prog, initial_val_constraint,
                    final_val_constraint, x, u, dt)
            else:
                r, u_optimal = calculate_r_duplicate(
                    ns, n0, n1, n2, prog, initial_val_constraint,
                    final_val_constraint, x, u, dt)

            # r(0,0) = 0
            eq_ind = np.argmin(np.linalg.norm(s, axis=0))
            r[eq_ind, eq_ind] = 0

            np.save(
                folder_name + "r_" +
                str(nT) +
                "_" +
                str(n_grid_xy) +
                "_" +
                str(n_grid_angle) +
                "_" +
                str(dt_max),
                r)
            np.save(
                folder_name + "u_" +
                str(nT) +
                "_" +
                str(n_grid_xy) +
                "_" +
                str(n_grid_angle) +
                "_" +
                str(dt_max),
                u_optimal)

        V, policy = value_iteration(ns, r, u_optimal, discount_factor=1)
        # Control at eq should be 0
        policy[eq_ind] = np.array([0, 0])

        np.save(
            folder_name + "V_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max),
            V)

        np.save(
            folder_name + "policy_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max),
            policy)

        np.save(
            folder_name + "s_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max),
            s)

    V_lambda = 0.5
    controller_relu = utils.setup_relu((3, 8, 8, 2),
                                       params=None,
                                       negative_slope=0.1,
                                       bias=True,
                                       dtype=torch.float64)
    lyapunov_relu = utils.setup_relu((3, 10, 10, 10, 1),
                                     params=None,
                                     negative_slope=0.01,
                                     bias=True,
                                     dtype=torch.float64)
    controller_lambda_u = 4
    controller_Ru = torch.tensor([[1, -1], [0, 1], [1, 0], [1, 1], [0.5, 0.9]],
                                 dtype=torch.float64)
    R = torch.cat((torch.eye(3, dtype=torch.float64),
                   torch.tensor([[1, -1, 0], [-1, -1, 1], [0, 1, 1]],
                                dtype=torch.float64)),
                  dim=0)

    if args.train_approximator:
        traj_opt_states = torch.from_numpy(s.T)
        # traj_opt_controls = torch.from_numpy(policy)
        traj_opt_costs = torch.from_numpy(V.T.reshape(-1, 1))
        train_unicycle_demo.train_cost_approximator(lyapunov_relu,
                                                    V_lambda,
                                                    R,
                                                    traj_opt_states,
                                                    traj_opt_costs,
                                                    num_epochs=100,
                                                    lr=0.001)
        utils.save_lyapunov_model(
            lyapunov_relu,
            V_lambda,
            0.,
            0.,
            None,
            r_options.FixedROptions(R),
            "neural_network_lyapunov/examples/car/data/cost_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max) +
            ".pt")
        dp_states, dp_actions = greedy_policy(s, a, lyapunov_relu, V_lambda, R, dt_max)
        dp_states = torch.from_numpy(dp_states.T)
        dp_actions = torch.from_numpy(dp_actions.T)
        train_unicycle_demo.train_controller_approximator(controller_relu,
                                                          dp_states,
                                                          dp_actions,
                                                          controller_lambda_u,
                                                          controller_Ru,
                                                          num_epochs=300,
                                                          lr=0.001)
        Ru_options = r_options.SearchRwithSVDOptions(controller_Ru.shape,
                                                     np.array([0.1, 0.2]))
        Ru_options.set_variable_value(controller_Ru.detach().numpy())
        train_unicycle_demo.save_controller_model(
            controller_relu,
            torch.from_numpy(x_lo),
            torch.from_numpy(x_up),
            torch.from_numpy(u_lo),
            torch.from_numpy(u_up),
            controller_lambda_u,
            Ru_options,
            "neural_network_lyapunov/examples/car/data/controller_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max) +
            ".pt")
    else:
        controller_data = torch.load(
                "neural_network_lyapunov/examples/car/data/controller_" +
                str(nT) +
                "_" +
                str(n_grid_xy) +
                "_" +
                str(n_grid_angle) +
                "_" +
                str(dt_max) +
                ".pt")

        controller_relu = utils.setup_relu(
            controller_data["linear_layer_width"],
            params=None,
            negative_slope=controller_data["negative_slope"],
            bias=controller_data["bias"],
            dtype=torch.float64)
        controller_relu.load_state_dict(controller_data["state_dict"])
        controller_lambda_u = controller_data["lambda_u"]
        controller_Ru = controller_data["Ru"]

        lyapunov_data = torch.load(
                "neural_network_lyapunov/examples/car/data/cost_" +
                str(nT) +
                "_" +
                str(n_grid_xy) +
                "_" +
                str(n_grid_angle) +
                "_" +
                str(dt_max) +
                ".pt")
        lyapunov_relu = utils.setup_relu(
            lyapunov_data["linear_layer_width"],
            params=None,
            negative_slope=lyapunov_data["negative_slope"],
            bias=lyapunov_data["bias"],
            dtype=torch.float64)
        lyapunov_relu.load_state_dict(lyapunov_data["state_dict"])
        V_lambda = lyapunov_data["V_lambda"]
        R = lyapunov_data["R"]

    plant = unicycle.Unicycle(torch.float64)
    x0 = np.array([0.5, 0.5, 0.5 * np.pi])  # Choose your initial state
    t_span = (0, 10)
    result = utils.\
        simulate_plant_with_controller(plant,
                                       controller_relu,
                                       t_span,
                                       torch.tensor([0, 0, 0],
                                                    dtype=torch.float64),
                                       torch.tensor([0, 0],
                                                    dtype=torch.float64),
                                       torch.from_numpy(u_lo),
                                       torch.from_numpy(u_up),
                                       x0)
    print(result.y[:, -1])
