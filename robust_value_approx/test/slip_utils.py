import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.slip_hybrid_linear_system as\
    slip_hybrid_linear_system
import robust_value_approx.spring_loaded_inverted_pendulum as\
    spring_loaded_inverted_pendulum
import robust_value_approx.hybrid_linear_system as hybrid_linear_system
import robust_value_approx.utils as utils

import torch
import numpy as np


def get_value_function(xf, N=3):
    assert(isinstance(xf, torch.Tensor))
    mass = 80
    l0 = 1
    gravity = 9.81
    dimensionless_spring_constant = 20.
    k = dimensionless_spring_constant * mass * gravity / l0
    # nonlinear system
    slip = spring_loaded_inverted_pendulum.SLIP(mass, l0, k, gravity)
    # piecewise linear system
    dtype = torch.float64
    x_lo = torch.Tensor([0., .1, 0.]).type(dtype)
    x_up = torch.Tensor([5., 1.5, 5.]).type(dtype)
    u_lo = torch.Tensor([np.pi/9]).type(dtype)
    u_up = torch.Tensor([np.pi/3]).type(dtype)
    num_breaks_x = [1, 5, 5]
    num_breaks_u = [15]
    u_scale_down = .45
    x_dim = len(num_breaks_x)
    u_dim = len(num_breaks_u)
    slip_hls = slip_hybrid_linear_system.SlipHybridLinearSystem(
        mass, l0, k, gravity)
    slip_hls.add_stepping_stone(x_lo[0], x_up[0], 0)
    all_limits = []
    all_samples = []
    indeces = []
    for i in range(len(x_lo)):
        lim_ = np.linspace(x_lo[i], x_up[i], num_breaks_x[i] + 1)
        limits = [(lim_[k], lim_[k+1]) for k in range(num_breaks_x[i])]
        samples = [.5*(limits[k][0] + limits[k][1])
                   for k in range(num_breaks_x[i])]
        all_limits.append(limits)
        all_samples.append(samples)
        indeces.append(np.arange(num_breaks_x[i]))
    for i in range(len(u_lo)):
        lim_ = np.linspace(u_lo[i], u_up[i], num_breaks_u[i] + 1)
        du = [lim_[k+1] - lim_[k] for k in range(num_breaks_u[i])]
        limits = [(lim_[k] + u_scale_down*du[k], lim_[k+1] -
                   u_scale_down*du[k]) for k in range(num_breaks_u[i])]
        samples = [.5*(limits[k][0] + limits[k][1])
                   for k in range(num_breaks_u[i])]
        all_limits.append(limits)
        all_samples.append(samples)
        indeces.append(np.arange(num_breaks_u[i]))
    grid = np.meshgrid(*indeces)
    indeces_samples = np.concatenate([g.reshape(-1, 1) for g in grid], axis=1)
    hls = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)
    for k in range(indeces_samples.shape[0]):
        s = indeces_samples[k, :]
        state = np.array([all_samples[i][s[i]] for i in range(x_dim+u_dim)])
        x = state[:x_dim]
        u = state[x_dim:x_dim+u_dim]
        xu_lim = np.array([all_limits[i][s[i]] for i in range(x_dim+u_dim)])
        xu_lo = xu_lim[:, 0]
        xu_up = xu_lim[:, 1]
        (A, B, c,
         a_t, b_t, c_t,
         P, q) = slip_hls.apex_map_linear_approximation(
            x.squeeze(),
            slip_hls.stepping_stones[0],
            u.squeeze())
        if A is not None:
            A = torch.Tensor(A).type(dtype)
            B = torch.Tensor(B).type(dtype)
            c = torch.Tensor(c).type(dtype).squeeze()
            P = torch.Tensor(P).type(dtype)
            q = torch.Tensor(q).type(dtype).squeeze()
            P = torch.cat((P, torch.eye(x_dim+u_dim, dtype=dtype)), axis=0)
            P = torch.cat((P, -torch.eye(x_dim+u_dim, dtype=dtype)), axis=0)
            q = torch.cat((q, torch.Tensor(xu_up).type(dtype), -
                           torch.Tensor(xu_lo).type(dtype)), axis=0)
            hls.add_mode(A, B, c, P, q, check_polyhedron_bounded=True)
        utils.update_progress((k + 1) / indeces_samples.shape[0])
    print(str(hls.num_modes) + " hybrid modes created")
    R = torch.eye(u_dim)
    Q = torch.diag(torch.Tensor([1., 1., 0.]).type(dtype))
    Rt = torch.eye(u_dim)
    Qt = torch.diag(torch.Tensor([100., 100., 0.]).type(dtype))
    vf = value_to_optimization.ValueFunction(hls, N, x_lo, x_up, u_lo, u_up)
    vf.set_cost(Q=Q, R=R)
    vf.set_terminal_cost(Qt=Qt, Rt=Rt)
    xtraj = xf.type(dtype).unsqueeze(1).repeat(1, N-1)
    vf.set_traj(xtraj=xtraj)
    return vf, slip
