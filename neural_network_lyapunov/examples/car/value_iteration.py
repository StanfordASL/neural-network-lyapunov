import numpy as np
import neural_network_lyapunov.examples.car.unicycle_traj_opt as\
    unicycle_traj_opt
import pydrake.solvers.mathematicalprogram as mp


def value_iteration(nx, r, epsilon=0.0001, discount_factor=0.95):
    """
    Value Iteration Algorithm.
    Args:
        nx: number of state in the environment.
        discount_factor: Gamma discount factor.
    Returns:
        V: the optimal value function.
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
    nT = 4
    u_lo = np.array([-3, -0.25 * np.pi])
    u_up = np.array([6, 0.25 * np.pi])
    x_lo = np.array([-3, -3, -1.2 * np.pi])
    x_up = np.array([3, 3, 1.2 * np.pi])
    dt_min = 0.001
    dt_max = 0.05
    prog, initial_val_constraint, final_val_constraint, x, u, dt =\
        unicycle_traj_opt.construct_traj_opt(
            nT, u_lo, u_up, dt_min, dt_max)
    n_grid = 100
    s = np.array(np.meshgrid(np.linspace(x_lo[0], x_up[0], n_grid),
                             np.linspace(x_lo[1], x_up[1], n_grid),
                             np.linspace(x_lo[2], x_up[2], n_grid)))
    s = s.reshape((3, n_grid**3))
    # u = np.array(np.meshgrid(np.linspace(u_lo[0], u_up[0], n_grid),
    #                          np.linspace(u_lo[1], u_up[1], n_grid)))
    # u = u.reshape((2, n_grid ** 2))
    ns = s.shape[1]
    # nu = u.shape[1]
    r = np.zeros((ns, ns))
    for i in range(ns):
        if i % 100 == 0:
            print("r iteration: ", i)
        for j in range(ns):
            initial_val_constraint.evaluator().set_bounds(s[:, i], s[:, i])
            final_val_constraint.evaluator().set_bounds(s[:, j], s[:, j])
            prog.SetInitialGuess(x, np.linspace(s[:, i], s[:, j], nT).T)
            prog.SetInitialGuess(dt, dt_max * np.ones((nT - 1,)))
            result = mp.Solve(prog)
            if result.is_success():
                r[i, j] = result.get_optimal_cost()
            else:
                r[i, j] = np.inf
    np.save("value/r", r)

    V = value_iteration(ns, r, discount_factor=1)
    np.save("value/V", V)
