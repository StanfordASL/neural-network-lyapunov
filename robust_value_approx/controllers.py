# -*- coding: utf-8 -*-
import robust_value_approx.value_to_optimization as value_to_optimization
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver
import torch
import copy
import numpy as np
import scipy
import jax
import control


def get_inifinite_horizon_ctrl(dyn_con, vf, ctrl_model):
    x_lo = vf.x_lo[0]
    x_up = vf.x_up[0]
    u_lo = vf.u_lo[0]
    u_up = vf.u_up[0]
    dt_lo = vf.dt_lo
    dt_up = vf.dt_up
    Q = vf.Q[0]
    x_desired = vf.x_desired[0]
    R = vf.R[0]
    x_dim = x_lo.shape[0]
    u_dim = u_lo.shape[0]
    dtype = x_lo.dtype
    prog = MathematicalProgram()
    x0 = prog.NewContinuousVariables(x_dim, "x0")
    u0 = prog.NewContinuousVariables(u_dim, "u0")
    dt0 = prog.NewContinuousVariables(1, "dt0")
    x1 = prog.NewContinuousVariables(x_dim, "x1")
    u1 = prog.NewContinuousVariables(u_dim, "u1")
    dt1 = prog.NewContinuousVariables(1, "dt1")
    x0_constraint = prog.AddBoundingBoxConstraint(x_lo, x_up, x0)
    prog.AddBoundingBoxConstraint(u_lo, u_up, u0)
    prog.AddBoundingBoxConstraint(dt_lo, dt_up, dt0)
    prog.AddBoundingBoxConstraint(x_lo, x_up, x1)
    prog.AddBoundingBoxConstraint(u_lo, u_up, u1)
    prog.AddBoundingBoxConstraint(dt_lo, dt_up, dt1)
    prog.AddConstraint(dyn_con,
        lb=np.zeros(x_dim), ub=np.zeros(x_dim),
        vars=np.concatenate((x0, u0, dt0, x1, u1, dt1)))
    if Q is not None:
        if x_desired is not None:
            prog.AddQuadraticErrorCost(
                Q=Q, x_desired=x_desired, vars=x0)
        else:
            prog.AddQuadraticCost(
                Q=Q, b=np.zeros(x_dim), c=0., vars=x0)
    if R is not None:
        prog.AddQuadraticCost(
            Q=R, b=np.zeros(u_dim), c=0., vars=u0)
    a_nn = prog.NewContinuousVariables(x_dim, "a_nn")
    b_nn = prog.NewContinuousVariables(1, "b_nn")
    a_nn_constraint = prog.AddBoundingBoxConstraint(
        np.zeros(x_dim), np.zeros(x_dim), a_nn)
    b_nn_constraint = prog.AddBoundingBoxConstraint(
        np.zeros(1), np.zeros(1), b_nn)
    def nn_cost(vars):
        a_nn_start = 0
        a_nn_end = a_nn_start + x_dim
        b_nn_start = a_nn_end
        b_nn_end = b_nn_start + 1
        x0_start = b_nn_end
        x0_end = x0_start + x_dim
        x1_start = x0_end
        x1_end = x1_start + x_dim
        a_nn = vars[a_nn_start:a_nn_end]
        b_nn = vars[b_nn_start:b_nn_end]
        x0 = vars[x0_start:x0_end]
        x1 = vars[x1_start:x1_end]
        return np.dot(a_nn, x1 - x0) + b_nn[0]
    prog.AddCost(nn_cost, vars=np.concatenate((a_nn, b_nn, x0, x1)))
    solver = SnoptSolver()
    def ctrl(x):
        assert(isinstance(x, torch.Tensor))
        dtype = x.dtype
        x.requires_grad = True
        b = ctrl_model(x)
        a = torch.autograd.grad(b, x)[0]
        b_np = b.detach().numpy()
        a_np = a.detach().numpy()
        a_nn_constraint.evaluator().set_bounds(a_np, a_np)
        b_nn_constraint.evaluator().set_bounds(b_np, b_np)
        x = x.detach().numpy()
        x0_constraint.evaluator().set_bounds(x, x)
        result = solver.Solve(prog, np.zeros(prog.num_vars()), None)
        if not result.is_success():
            return(None, None, None)
        u0_opt = result.GetSolution(u0)
        u1_opt = result.GetSolution(u1)
        x_opt = result.GetSolution(x1)
        return(torch.Tensor(u0_opt).type(dtype),
            torch.Tensor(u1_opt).type(dtype),
            torch.Tensor(x_opt).type(dtype))
    return ctrl


def get_sampling_infinite_horizon_controller(dx, step_cost, ctrl_model,
                                             u_min, u_max, dt=.2,
                                             num_samples=50):
    u_dim = u_min.shape[0]
    dtype = u_min.dtype
    u_min_np = u_min.detach().numpy()
    u_max_np = u_max.detach().numpy()
    def ctrl(x):
        x0_np = x.detach().numpy()
        v_opt = float("Inf")
        u_opt = None
        for k in range(num_samples):
            u = np.random.rand(u_dim) * (u_max_np - u_min_np) + u_min_np
            sim_dyn = lambda t, y: dx(y, u)
            traj = scipy.integrate.solve_ivp(sim_dyn, (0, dt), x0_np)
            if traj.success:
                cost = step_cost(0, x, torch.Tensor(u).type(dtype))
                xn = torch.Tensor(traj.y[:,-1]).type(dtype)
                v = cost + torch.clamp(ctrl_model(xn), 0.)
                if v < v_opt:
                    v_opt = v
                    u_opt = u
        u_opt = torch.Tensor(u_opt).type(dtype)
        return (u_opt, u_opt, None)
    return ctrl


def get_optimal_controller(vf):
    assert(isinstance(vf, value_to_optimization.NLPValueFunction))
    assert(vf.x0_constraint is not None)
    def ctrl(x):
        assert(isinstance(x, torch.Tensor))
        dtype = x.dtype
        x = x.detach().numpy()
        vf.x0_constraint.evaluator().set_bounds(x, x)
        result = vf.solver.Solve(
            vf.prog, np.zeros(vf.prog.num_vars()), None)
        if not result.is_success():
            return(None, None, None)
        u0 = result.GetSolution(vf.u_traj[0])
        u1 = result.GetSolution(vf.u_traj[1])
        x_opt = result.GetSolution(vf.x_traj[1])
        return(torch.Tensor(u0).type(dtype),
            torch.Tensor(u1).type(dtype),
            torch.Tensor(x_opt).type(dtype))
    return ctrl


def get_lqr_controller(dx, x0, u0, Q, R, u_min, u_max):
    x_dim = x0.shape[0]
    u_dim = u0.shape[0]
    dtype = x0.dtype
    J = jax.jacobian(
        lambda x: dx(x[:x_dim], x[x_dim:x_dim+u_dim], arraylib=jax.numpy))
    AB = J(torch.cat((x0, u0)).detach().numpy())
    A = np.array(AB[:, :x_dim])
    B = np.array(AB[:, x_dim:x_dim+u_dim])
    if isinstance(Q, torch.Tensor):
        Q = Q.detach().numpy()
    if isinstance(R, torch.Tensor):
        R = R.detach().numpy()
    K, S, E = control.lqr(A, B, Q, R)
    K = torch.Tensor(K).type(dtype)
    def ctrl(x):
        assert(isinstance(x, torch.Tensor))  
        u = -K@(x - x0) + u0
        u = torch.max(torch.min(u, u_max), u_min)
        return(u, u, None)
    return ctrl


def sim_ctrl(x0, u_dim, dx, ctrl, dt, N):
    assert(isinstance(x0, torch.Tensor))
    dtype = x0.dtype
    x_traj = [x0.unsqueeze(1).detach().numpy()]
    t_traj = [np.array([0.])]
    for n in range(N-1):
        x0_np = x_traj[-1][:,-1]
        t0 = t_traj[-1][-1]
        u0, u1, _ = ctrl(torch.Tensor(x0_np).type(dtype))
        if u0 is not None:
            u0 = u0.detach().numpy()
            u1 = u1.detach().numpy()
        def sim_dyn(t, y):
            if u0 is not None:
                s = (t - t0)/dt
                u = (1 - s)*u0 + s*u1
            else:
                u = np.zeros(u_dim)
            return dx(y, u)
        traj = scipy.integrate.solve_ivp(sim_dyn, (t0, t0+dt), x0_np)
        if not traj.success:
            print("Warning: simulation failed")
            break
        x_traj.append(traj.y[:,1:])
        t_traj.append(traj.t[1:])
    x_traj = np.concatenate(x_traj, axis=1)
    t_traj = np.concatenate(t_traj)
    return(torch.Tensor(x_traj).type(x0.dtype),
        torch.Tensor(t_traj).type(x0.dtype))


def benchmark_controller(u_dim, dx, ctrl, x0, x0_eps, num_breaks, x_goal, 
                         dt, N, dim1=0, dim2=1):
    dtype = x0.dtype
    x_dim1 = torch.linspace(x0[dim1] - x0_eps[dim1],
        x0[dim1] + x0_eps[dim1], num_breaks[0])
    x_dim2 = torch.linspace(x0[dim2] - x0_eps[dim2],
        x0[dim2] + x0_eps[dim2], num_breaks[1])
    bench = torch.zeros((num_breaks[0], num_breaks[1]), dtype=dtype)
    for i in range(len(x_dim1)):
        for j in range(len(x_dim2)):
            x0_ = x0.clone()
            x0_[dim1] = x_dim1[i]
            x0_[dim2] = x_dim2[j]
            x_traj_sim, t_traj_sim = sim_ctrl(x0_, u_dim, dx, ctrl, dt, N)
            xf = x_traj_sim[:, -1]
            bench[i,j] = torch.norm(xf - x_goal).item()
    return bench