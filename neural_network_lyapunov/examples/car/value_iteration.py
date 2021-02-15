import numpy as np
import neural_network_lyapunov.examples.car.unicycle_traj_opt as\
    unicycle_traj_opt
import pydrake.solvers.mathematicalprogram as mp
import math
import argparse


def value_iteration(nx, r, epsilon=0.0001, discount_factor=0.95):
    """
    Value Iteration Algorithm.
    Args:
        nx: number of state in the environment.
        r: numpy array of size (nx, nx). immediate cost from state x to x'.
        epsilon: threshold for convergence.
        discount_factor: Gamma discount factor.
    Returns:
        V: numpy array of size nx. V[i] is the optimal value function of
        state s[i].
    """

    def one_step_lookahead(V, r, i):
        return np.min(r[i, :] + discount_factor * V)

    # start with inital value function and intial policy

    V = np.zeros(nx)

    n = 0
    # while not the optimal policy
    while True:
        print('Iteration: ', n)
        # for stopping condition
        delta = 0

        # loop over state space
        for i in range(nx):

            v = one_step_lookahead(V, r, i)

            # get the biggest difference between best action value and our old
            # value function
            if v != V[i]:
                delta = max(delta, abs(v - V[i]))

            # apply bellman optimality eqn
            V[i] = v

        # if optimal value function
        if (delta < epsilon):
            break
        n += 1

    return V


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
    # r = np.ones((ns, ns))*1000
    r = np.full((ns, ns), np.inf)
    u_optimal = np.full((ns, ns, 2), np.nan)
    for i in range(10):
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
    r = np.full((ns, ns), np.inf)
    u_optimal = np.zeros((ns, ns))
    # Calculate a quarter of a duplicated block at the four corners
    i0s = [0, 0, n_grid_xy - 1, n_grid_xy - 1]
    i1s = [0, n_grid_xy - 1, 0, n_grid_xy - 1]
    for i2 in range(n_grid_angle):
        # Construct duplicated 3D block
        r3d = np.full((2 * n0 + 1, 2 * n1 + 1, 2 * n2 + 1), np.inf)
        u3d = np.zeros((2 * n0 + 1, 2 * n1 + 1, 2 * n2 + 1))
        for k in range(len(i0s)):
            i0 = i0s[k]
            i1 = i1s[k]
            i = threeD_to_flat(i0, i1, i2)

            for j in range(ns):
                j0, j1, j2 = flat_to_3d(j)
                if j % 1000 == 0:
                    print("Duplicate iteration: ", j)
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
                        optimal_control = result.GetSolution(u)[0]
                        u3d[j0 - i0 + n0, j1 - i1 + n1,
                            j2 - i2 + n2] = optimal_control
                        u_optimal[i, j] = optimal_control

    for i in range(ns):
        i0, i1, i2 = flat_to_3d(i)
        if i % 100 == 0:
            print("r iteration: ", i)
        for j in range(ns):
            j0, j1, j2 = flat_to_3d(j)
            if i != j and 0 <= j0 - i0 + \
                    n0 < r3d.shape[0] and \
                    0 <= j1 - i1 + n1 < r3d.shape[1] and \
                    0 <= j2 - i2 + n2 < r3d.shape[2]:
                r[i, j] = r3d[j0 - i0 + n0, j1 - i1 + n1, j2 - i2 + n2]

    return r


def flat_to_3d(i):
    i0 = i // (n_grid_angle * n_grid_xy)
    i1 = (i - n_grid_angle * n_grid_xy * i0) // n_grid_angle
    i2 = i - n_grid_angle * n_grid_xy * i0 - n_grid_angle * i1
    return i0, i1, i2


def threeD_to_flat(i0, i1, i2):
    return n_grid_angle * n_grid_xy * i0 + n_grid_angle * i1 + i2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_r",
        help="load precomputed r ",
        action="store_true")
    parser.add_argument(
        "--load_V",
        help="load computed V ",
        action="store_true")
    args = parser.parse_args()

    epsilon = 1e-4

    nT = 5
    u_lo = np.array([-3, -0.25 * np.pi])
    u_up = np.array([6, 0.25 * np.pi])
    x_lo = np.array([-1, -1, -1.05 * np.pi])
    x_up = np.array([1, 1, 1.05 * np.pi])
    dt_min = 0.001
    dt_max = 0.07
    prog, initial_val_constraint, final_val_constraint, x, u, dt =\
        unicycle_traj_opt.construct_traj_opt(
            nT, u_lo, u_up, dt_min, dt_max)
    n_grid_xy = 10
    n_grid_angle = 30
    s = np.array(np.meshgrid(np.linspace(x_lo[0], x_up[0], n_grid_xy),
                             np.linspace(x_lo[1], x_up[1], n_grid_xy),
                             np.linspace(x_lo[2], x_up[2], n_grid_angle)))
    # ss = s.copy()
    s = s.reshape((3, n_grid_xy * n_grid_xy * n_grid_angle))
    ns = s.shape[1]
    n0 = math.ceil(u_up[0] * dt_max * (nT - 1) /
                   (x_up[0] - x_lo[0]) * n_grid_xy)
    n1 = math.ceil(u_up[0] * dt_max * (nT - 1) /
                   (x_up[1] - x_lo[1]) * n_grid_xy)
    n2 = math.ceil(u_up[1] * dt_max * (nT - 1) /
                   (x_up[2] - x_lo[2]) * n_grid_angle)
    if args.load_r:
        r = np.load(
            "neural_network_lyapunov/examples/car/value/r_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max) +
            ".npy")
    elif args.load_V:
        V = np.load(
            "neural_network_lyapunov/examples/car/value/V_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max) +
            ".npy")
    else:
        r, u_optimal = calculate_r_complete(
            ns, n0, n1, n2, prog, initial_val_constraint,
            final_val_constraint, x, u, dt)
        r = calculate_r_duplicate(ns, n0, n1, n2,
                                  prog, initial_val_constraint,
                                  final_val_constraint, x, u, dt)
        # r(0,0) = 0
        eq_ind = np.argmin(np.linalg.norm(s, axis=0))
        r[eq_ind, eq_ind] = 0
        np.save(
            "neural_network_lyapunov/examples/car/value/r_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max),
            r)
        np.save(
            "neural_network_lyapunov/examples/car/value/u_" +
            str(nT) +
            "_" +
            str(n_grid_xy) +
            "_" +
            str(n_grid_angle) +
            "_" +
            str(dt_max),
            u_optimal)

    V = value_iteration(ns, r, discount_factor=1)
    np.save(
        "neural_network_lyapunov/examples/car/value/V_" +
        str(nT) +
        "_" +
        str(n_grid_xy) +
        "_" +
        str(n_grid_angle) +
        "_" +
        str(dt_max),
        V)
    np.save(
        "neural_network_lyapunov/examples/car/value/s_" +
        str(nT) +
        "_" +
        str(n_grid_xy) +
        "_" +
        str(n_grid_angle) +
        "_" +
        str(dt_max),
        s)
