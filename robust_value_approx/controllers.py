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
import plotly
import plotly.graph_objs as go


def eigenautodiff_vf_approx(vf_approx, x_numpy, first_is_time=False):
    if first_is_time:
        x_numpy[0] = np.maximum(x_numpy[0], 0.)
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
        if x_numpy[0] <= 0:
            y_numpy *= 0.
        return y_numpy
    else:
        # x is an eigen vector of doubles
        if x_numpy[0] <= 0:
            return np.zeros(1)
        return torch.clamp(
            vf_approx.eval(torch.from_numpy(x_numpy)), 0.).detach().numpy()


def get_learned_policy_controller(policy_approx, final_time=None, dt=None):
    assert(isinstance(policy_approx, value_approximation.FunctionApproximation))
    if final_time is not None:
        assert(dt is not None)
        def ctrl(t, x):



            t = 0.
            


            assert(isinstance(t, float))
            assert(isinstance(x, torch.Tensor))
            t_to_go0 = max(final_time - t, 0.)
            u0 = policy_approx.eval(torch.cat([torch.tensor([t_to_go0],
                dtype=policy_approx.dtype), x]))
            # TODO: u1 should use x1, which is integrated from x0 and u0
            t_to_go1 = max(final_time - t - dt, 0.)
            u1 = policy_approx.eval(torch.cat([torch.tensor([t_to_go1],
                dtype=policy_approx.dtype), x]))
            return(u0, u1, None)
    else:
        raise(NotImplementedError)
    return ctrl


def get_limited_lookahead_controller(vf, vf_approx=None, final_time=None):
    """
    WARNING this function modifies the value function!!!
    """
    assert(isinstance(vf, value_to_optimization.ValueFunction))
    if vf_approx is not None:
        assert(isinstance(
            vf_approx, value_approximation.FunctionApproximation))
        if final_time is not None:
            t0 = vf.prog.NewContinuousVariables(1, "t0")
            t0_con = vf.prog.AddBoundingBoxConstraint(0, 0, t0)
            t_to_go = vf.prog.NewContinuousVariables(1, "t_to_go")
            t_to_go_con = vf.prog.AddConstraint(
                (final_time - t0 - np.sum(vf.dt_traj) - t_to_go)[0] == 0.)
            xf = vf.x_traj[-1]
            vf.prog.AddCost(lambda x: eigenautodiff_vf_approx(vf_approx, x,
                first_is_time=True)[0], vars=np.concatenate([t_to_go, xf]))
            V = vf.get_value_function()
            def ctrl(t, x):
                assert(isinstance(x, torch.Tensor))
                t0_con.evaluator().set_bounds(np.array([t]), np.array([t]))
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
        else:
            xf = vf.x_traj[-1]
            vf.prog.AddCost(lambda x: eigenautodiff_vf_approx(vf_approx, x,
                first_is_time=False)[0], vars=xf)
            V = vf.get_value_function()
            def ctrl(t, x):
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
    else:
        V = vf.get_value_function()
        def ctrl(t, x):
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
    def ctrl(t, x):
        assert(isinstance(x, torch.Tensor))  
        u = -K@(x - x0) + u0
        u = torch.max(torch.min(u, u_max), u_min)
        return(u, u, None)
    return ctrl, S


def sim_ctrl(x0, u_dim, dx, ctrl, dt, N, integration_mode="foh"):
    assert(isinstance(x0, torch.Tensor))
    dtype = x0.dtype
    x_traj = [x0.unsqueeze(1).detach().numpy()]
    t_traj = [np.array([0.])]
    for n in range(N-1):
        x0_np = x_traj[-1][:,-1]
        t0 = t_traj[-1][-1]
        u0, u1, _ = ctrl(t0, torch.Tensor(x0_np).type(dtype))
        if u0 is not None:
            u0 = u0.detach().numpy()
            u1 = u1.detach().numpy()
        else:
            u0 = np.zeros(u_dim)
            u1 = np.zeros(u_dim)
        if integration_mode == "backward":
            u0 = np.array(u1)
        elif integration_mode == "forward":
            u1 = np.array(u0)
        elif integration_mode == "foh":
            pass
        else:
            raise(NotImplementedError)
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
                         dt, N, dim1=0, dim2=1, integration_mode="foh"):
    dtype = x0.dtype
    x_dim1 = torch.linspace(x0[dim1] - x0_eps[dim1],
        x0[dim1] + x0_eps[dim1], num_breaks[dim1])
    x_dim2 = torch.linspace(x0[dim2] - x0_eps[dim2],
        x0[dim2] + x0_eps[dim2], num_breaks[dim2])
    bench = torch.zeros((num_breaks[dim1], num_breaks[dim2]), dtype=dtype)
    for i in range(len(x_dim1)):
        for j in range(len(x_dim2)):
            x0_ = x0.clone()
            x0_[dim1] = x_dim1[i]
            x0_[dim2] = x_dim2[j]
            x_traj_sim, t_traj_sim = sim_ctrl(
                x0_, u_dim, dx, ctrl, dt, N, integration_mode)
            xf = x_traj_sim[:, -1]
            bench[i,j] = torch.norm(xf - x_goal).item()
    return bench


def plot_sim(t_traj, x_traj, title=""):
    fig = go.Figure()
    for i in range(x_traj.shape[0]):
        fig.add_trace(go.Scatter(
            x=t_traj,
            y=x_traj[i,:]
        ))
    fig.update_layout(
        title=title,
    )
    return fig