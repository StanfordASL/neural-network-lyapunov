"""
Consider a integrator xdot = u, -1 <= u <= 1. Train a barrier function.
"""
import neural_network_lyapunov.control_affine_system as control_affine_system
import neural_network_lyapunov.barrier as barrier
import neural_network_lyapunov.control_barrier as control_barrier
import neural_network_lyapunov.train_barrier as train_barrier
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import gurobipy
import scipy.integrate


def simulate(barrier_relu, x_star, c, epsilon, x0, T):
    def dynamics(t, x):
        prog = gurobipy.Model()
        u_var = prog.addVar(lb=-1, ub=1)
        x_torch = torch.from_numpy(x)
        dhdx = utils.relu_network_gradient(barrier_relu, x_torch).squeeze(1)
        h_val = barrier_relu(x_torch) - barrier_relu(x_star) + c
        for i in range(dhdx.shape[0]):
            prog.addLConstr(gurobipy.LinExpr(dhdx[i], [u_var]),
                            gurobipy.GRB.GREATER_EQUAL,
                            -epsilon * h_val.item())
        prog.setObjective(u_var * u_var, gurobipy.GRB.MINIMIZE)
        prog.setParam(gurobipy.GRB.Param.OutputFlag, False)
        prog.optimize()
        assert (prog.status == gurobipy.GRB.Status.OPTIMAL)
        u_val = np.array([u_var.x])
        return u_val

    return scipy.integrate.solve_ivp(dynamics, [0, T], x0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="single integrator cbf training")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    dtype = torch.float64
    x_lo = torch.tensor([-1], dtype=dtype)
    x_up = torch.tensor([1], dtype=dtype)
    u_lo = torch.tensor([-1], dtype=dtype)
    u_up = torch.tensor([1], dtype=dtype)
    system = control_affine_system.LinearSystem(A=torch.tensor([[0]],
                                                               dtype=dtype),
                                                B=torch.tensor([[1]],
                                                               dtype=dtype),
                                                x_lo=x_lo,
                                                x_up=x_up,
                                                u_lo=u_lo,
                                                u_up=u_up)
    barrier_relu = utils.setup_relu((1, 4, 2, 1),
                                    params=None,
                                    negative_slope=0.1,
                                    bias=True)
    x_star = torch.tensor([0], dtype=dtype)
    c = 0.5
    barrier_system = control_barrier.ControlBarrier(system, barrier_relu)

    verify_region_boundary = utils.box_boundary(x_lo, x_up)
    unsafe_region_cnstr = gurobi_torch_mip.MixedIntegerConstraintsReturn()
    epsilon = 0.1
    inf_norm_term = barrier.InfNormTerm(
        torch.diag(2. / (system.x_up - system.x_lo)),
        (system.x_up + system.x_lo) / (system.x_up - system.x_lo))
    dut = train_barrier.TrainBarrier(barrier_system, x_star, c,
                                     unsafe_region_cnstr,
                                     verify_region_boundary, epsilon,
                                     inf_norm_term)

    unsafe_state_samples = torch.zeros((0, 1), dtype=dtype)
    boundary_state_samples = torch.zeros((0, 1), dtype=dtype)
    deriv_state_samples = torch.zeros((0, 1), dtype=dtype)
    is_success = dut.train(unsafe_state_samples, boundary_state_samples,
                           deriv_state_samples)
    if args.plot:
        with torch.no_grad():
            x_samples = torch.linspace(x_lo.item(),
                                       x_up.item(),
                                       100,
                                       dtype=dtype)
            h_samples = barrier_system.barrier_value(x_samples.unsqueeze(1),
                                                     x_star, c).squeeze()
            hdot_samples = torch.stack([
                torch.min(
                    barrier_system.barrier_derivative(
                        torch.tensor([x_samples[i]], dtype=dtype)))
                for i in range(x_samples.shape[0])
            ])
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.plot(x_samples.detach().numpy(), h_samples.detach().numpy())
        ax1.set_xlabel("x")
        ax1.set_ylabel("h")
        ax1.set_title("barrier function")
        ax2 = fig.add_subplot(312)
        ax2.plot(x_samples.detach().numpy(), hdot_samples.detach().numpy())
        ax2.set_xlabel("x")
        ax2.set_ylabel(r"$\dot{h}$")
        ax3 = fig.add_subplot(313)
        ax3.plot(x_samples.detach().numpy(),
                 (hdot_samples + epsilon * h_samples).detach().numpy())
        ax3.set_xlabel("x")
        ax3.set_ylabel(r"$\dot{h}+\epsilon h$")
        fig.show()

    assert (is_success)
    sim_result = simulate(barrier_relu,
                          x_star,
                          c,
                          epsilon,
                          np.array([0.]),
                          T=10)
    pass
