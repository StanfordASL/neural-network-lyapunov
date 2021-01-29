import pydrake.solvers.mathematicalprogram as mp
import neural_network_lyapunov.examples.car.unicycle as unicycle
import neural_network_lyapunov.utils as utils

import numpy as np
import torch


def dynamics_constraint_evaluator(plant, xut):
    x_curr = xut[:3]
    x_next = xut[3:6]
    u_curr = xut[6:8]
    u_next = xut[8:10]
    delta_t = xut[-1]
    xdot_curr = plant.dynamics(x_curr, u_curr)
    xdot_next = plant.dynamics(x_next, u_next)
    return x_next - x_curr - (xdot_next + xdot_curr) / 2 * delta_t


def cost_trajectory(nT, ut):
    u = ut[:2 * nT].reshape((2, nT))
    dt_traj = ut[2 * nT:3 * nT - 1]
    return ((u[0, :-1] + u[0, 1:]) / 2)**2 * dt_traj + (
        (u[1, :-1] + u[1, 1:]) / 2)**2 * dt_traj


def cost(nT, ut):
    return np.sum(cost_trajectory(nT, ut))


def construct_traj_opt(nT, u_lo, u_up, dt_min, dt_max):
    """
    Construct a trajectory optimization problem to find the minimal time to
    reach the goal position. The user should adjusts initial_val_constraint to
    set the initial state correctly.
    """
    plant = unicycle.Unicycle(torch.float64)
    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(3, nT)
    u = prog.NewContinuousVariables(2, nT)
    dt = prog.NewContinuousVariables(nT - 1)
    prog.AddBoundingBoxConstraint(dt_min, dt_max, dt)
    for i in range(nT - 1):
        prog.AddConstraint(
            lambda xut: dynamics_constraint_evaluator(plant, xut),
            np.zeros((3, )), np.zeros((3, )),
            np.hstack((x[:, i], x[:, i + 1], u[:, i], u[:, i + 1], dt[i])))
    prog.AddCost(lambda ut: cost(nT, ut), np.hstack((u.reshape((-1, )), dt)))
    prog.AddBoundingBoxConstraint(u_lo[0], u_up[0], u[0, :])
    prog.AddBoundingBoxConstraint(u_lo[1], u_up[1], u[1, :])
    final_val_constraint = prog.AddBoundingBoxConstraint(
        np.zeros((3, )), np.zeros((3, )), x[:, -1])
    initial_val_constraint = prog.AddBoundingBoxConstraint(
        np.zeros((3, )), np.zeros((3, )), x[:, 0])
    return prog, initial_val_constraint, final_val_constraint, x, u, dt


def construct_training_set():
    u_lo = np.array([-3, -0.25 * np.pi])
    u_up = np.array([6, 0.25 * np.pi])
    dt_min = 0.001
    dt_max = 0.03
    nT = 40
    prog, initial_val_constraint, final_val_constraint, x, u, dt =\
        construct_traj_opt(nT, u_lo, u_up, dt_min, dt_max)
    # We sample many initial states in the box x_lo <= x <= x_up
    x_lo = torch.tensor([-3, -3, -1.2 * np.pi], dtype=torch.float64)
    x_up = torch.tensor([3, 3, 1.2 * np.pi], dtype=torch.float64)
    x_init_samples = utils.uniform_sample_in_box(x_lo, x_up, 3000)

    states = [torch.zeros((1, 3), dtype=torch.float64)]
    controls = [torch.zeros((1, 2), dtype=torch.float64)]
    costs = [torch.zeros((1, 1), dtype=torch.float64)]
    for i in range(x_init_samples.shape[0]):
        x_initial_state = x_init_samples[i].detach().numpy()
        initial_val_constraint.evaluator().set_bounds(x_initial_state,
                                                      x_initial_state)
        prog.SetInitialGuess(dt, dt_max * np.ones((nT - 1)))
        prog.SetInitialGuess(
            x,
            np.linspace(x_initial_state, np.zeros((3, )), nT).T)
        result = mp.Solve(prog)
        if result.is_success():
            x_traj = result.GetSolution(x)
            u_traj = result.GetSolution(u)
            dt_traj = result.GetSolution(dt)
            cost_traj = np.cumsum(
                cost_trajectory(nT, np.hstack((u_traj.reshape(
                    (-1, )), dt_traj)))[::-1])[::-1].copy()
            states.append(torch.from_numpy(x_traj[:, :-1].T))
            controls.append(torch.from_numpy(u_traj[:, :-1].T))
            costs.append(torch.from_numpy(cost_traj).reshape((-1, 1)))
    pass
    return states, controls, costs


if __name__ == "__main__":
    states, controls, costs = construct_training_set()
