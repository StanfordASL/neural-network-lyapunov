# -*- coding: utf-8 -*-
import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.value_approximation as value_approximation
from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver
from pydrake.autodiffutils import (autoDiffToValueMatrix,
    autoDiffToGradientMatrix, initializeAutoDiffGivenGradientMatrix)
import torch
import copy
import numpy as np
import scipy
import jax


def get_sampling_infinite_horizon_controller(dx, step_cost, ctrl_model,
                                             x_lo, x_up,
                                             u_lo, u_up,
                                             dt, num_samples):
    u_dim = u_lo.shape[0]
    dtype = u_lo.dtype
    u_lo_np = u_lo.detach().numpy()
    u_up_np = u_up.detach().numpy()
    def ctrl(x):
        x0_np = x.detach().numpy()
        v_opt = float("Inf")
        u_opt = None
        for k in range(num_samples):
            u = np.random.rand(u_dim) * (u_up_np - u_lo_np) + u_lo_np
            sim_dyn = lambda t, y: dx(y, u)
            traj = scipy.integrate.solve_ivp(sim_dyn, (0, dt), x0_np)
            if traj.success:
                xn = torch.Tensor(traj.y[:,-1]).type(dtype)
                if torch.all(xn < x_up) and torch.all(xn > x_lo):
                    u = torch.Tensor(u).type(dtype)
                    cost = step_cost(0, x, u, dt)
                    v = cost + torch.clamp(ctrl_model(xn), 0.)
                    if v < v_opt:
                        v_opt = v
                        u_opt = u
        return (u_opt, u_opt, None)
    return ctrl


def eigenautodiff_vf_approx(vf_approx, x_numpy):
    if x_numpy.dtype == np.object:
        # x_numpy is an numpy array of autodiff scalars.
        x_val = autoDiffToValueMatrix(x_numpy)
        dx_dz = autoDiffToGradientMatrix(x_numpy)
        x_torch = torch.from_numpy(x_val).squeeze()
        x_torch.requires_grad=True
        y = torch.clamp(vf_approx.eval(x_torch), 0.)
        y.backward()
        dy_dx = x_torch.grad.clone().detach().numpy()
        dy_dz = dy_dx @ dx_dz
        y_numpy = initializeAutoDiffGivenGradientMatrix(
            y.detach().numpy().reshape(1,-1), dy_dz.reshape(1,-1))
        return y_numpy
    else:
        # x is an eigen vector of doubles
        return torch.clamp(vf_approx.eval(torch.from_numpy(x_numpy)), 0.).detach().numpy()

def get_limited_lookahead_controller(vf, vf_approx=None):
    assert(isinstance(vf, value_to_optimization.ValueFunction))
    if vf_approx is not None:
        assert(isinstance(
            vf_approx, value_approximation.ValueFunctionApproximation))
        xf = vf.x_traj[-1]
        vf.prog.AddCost(
            lambda x: eigenautodiff_vf_approx(vf_approx, x)[0], vars=xf)
    V = vf.get_value_function()
    def ctrl(x):
        assert(isinstance(x, torch.Tensor))
        v, res = V(x)
        if v is not None:
            u0 = res['u_traj'][0]
            u1 = res['u_traj'][1]
            x1 = res['x_traj'][1]
        else:
            u0 = None
            u1 = None
            x1 = None
        return(u0, u1, x1)
    return ctrl


def lqr(A, B, Q, R):
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    return K, X, eigVals


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
    K, S, E = lqr(A, B, Q, R)
    K = torch.Tensor(K).type(dtype)
    def ctrl(x):
        assert(isinstance(x, torch.Tensor))  
        u = -K@(x - x0) + u0
        u = torch.max(torch.min(u, u_max), u_min)
        return(u, u, None)
    return ctrl, S


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
        else:
            u0 = np.zeros(u_dim)
            u1 = np.zeros(u_dim)
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