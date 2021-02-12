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
            delta = max(delta, abs(v - V[i]))

            # apply bellman optimality eqn
            V[i] = v

        # if optimal value function
        if (delta < epsilon):
            break
        n += 1

    return V


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_r",
        help="load precomputed r ",
        action="store_true")
    args = parser.parse_args()

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
    n_grid_xy = 20
    n_grid_angle = 60
    s = np.array(np.meshgrid(np.linspace(x_lo[0], x_up[0], n_grid_xy),
                             np.linspace(x_lo[1], x_up[1], n_grid_xy),
                             np.linspace(x_lo[2], x_up[2], n_grid_angle)))
    s = s.reshape((3, n_grid_xy * n_grid_xy * n_grid_angle))
    # u = np.array(np.meshgrid(np.linspace(u_lo[0], u_up[0], n_grid),
    #                          np.linspace(u_lo[1], u_up[1], n_grid)))
    # u = u.reshape((2, n_grid ** 2))
    ns = s.shape[1]
    # nu = u.shape[1]
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
            str(dt_max))
    else:
        r = np.ones((ns, ns))*1000
        # r = np.full((ns, ns), np.inf)
        for i in range(ns):
            i0 = i // (n_grid_angle * n_grid_xy)
            i1 = (i - n_grid_angle * n_grid_xy * i0) // n_grid_angle
            i2 = i - n_grid_angle * n_grid_xy * i0 - n_grid_angle * i1
            if i % 100 == 0:
                print("r iteration: ", i)
            for j in range(ns):
                j0 = j // n_grid_xy ** 2
                j1 = (j - n_grid_xy ** 2 * j0) // n_grid_xy
                j2 = j - n_grid_xy ** 2 * j0 - n_grid_xy * j1
                if (i0 - n0) < j0 < (i0 + n0) and\
                   (i1 - n1) < j1 < (i1 + n1) and\
                   (i2 - n2) < j2 < (i2 + n2):
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
                    else:
                        r[i, j] = np.inf
                else:
                    r[i, j] = np.inf
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
