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
    x_lo = torch.Tensor([0., .5, 2.5]).type(dtype)
    x_up = torch.Tensor([100., 1.5, 10.]).type(dtype)
    u_lo = torch.Tensor([np.pi/9]).type(dtype)
    u_up = torch.Tensor([np.pi/3]).type(dtype)
    num_breaks_x = [1, 5, 5]
    num_breaks_u = [5]
    # reduces the validity of the control input discretization
    u_scale_down = .475
    # increases the validity of the state space discretization
    x_buff = np.array([0., .05, .05])
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
        xu_lo[:x_dim] = np.minimum(np.maximum(
            xu_lo[:x_dim] - x_buff, x_lo), x_up)
        xu_up[:x_dim] = np.minimum(np.maximum(
            xu_up[:x_dim] + x_buff, x_lo), x_up)
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
    # cost function
    Q = torch.diag(torch.Tensor([1., 1., 0.]).type(dtype))
    Qt = torch.diag(torch.Tensor([100., 100., 0.]).type(dtype))
    vf = value_to_optimization.ValueFunction(hls, N, x_lo, x_up, u_lo, u_up)
    vf.set_cost(Q=Q)
    vf.set_terminal_cost(Qt=Qt)
    xtraj = xf.type(dtype).unsqueeze(1).repeat(1, N-1)
    vf.set_traj(xtraj=xtraj)
    return vf, slip


def get_value_function_gait(xf, N=2):
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
    x_lo = torch.Tensor([0., .5, 2.5]).type(dtype)
    x_up = torch.Tensor([100., 1.5, 10.]).type(dtype)
    u_lo = torch.Tensor([np.pi/9]).type(dtype)
    u_up = torch.Tensor([np.pi/3]).type(dtype)
    num_breaks_x = [1, 5, 5]
    num_breaks_u = [5]
    # reduces the validity of the control input discretization
    u_scale_down = .475
    # increases the validity of the state space discretization
    x_buff = np.array([0., .05, .05])
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
        xu_lo[:x_dim] = np.minimum(np.maximum(
            xu_lo[:x_dim] - x_buff, x_lo), x_up)
        xu_up[:x_dim] = np.minimum(np.maximum(
            xu_up[:x_dim] + x_buff, x_lo), x_up)
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
    # cost function
    Q = torch.diag(torch.Tensor([0., 1., 1.]).type(dtype))
    Qt = torch.diag(torch.Tensor([0., 100., 100.]).type(dtype))
    vf = value_to_optimization.ValueFunction(hls, N, x_lo, x_up, u_lo, u_up)
    vf.set_cost(Q=Q)
    vf.set_terminal_cost(Qt=Qt)
    xtraj = xf.type(dtype).unsqueeze(1).repeat(1, N-1)
    vf.set_traj(xtraj=xtraj)
    return vf, slip


def sim_slip(slip, x0, u_traj):
    assert(isinstance(x0, torch.Tensor))
    assert(isinstance(u_traj, torch.Tensor))
    theta_step = u_traj.squeeze().numpy()
    slip_apex_x = [x0[0]]
    slip_apex_y = [x0[1]]
    slip_apex_xdot = [x0[2]]
    for theta in theta_step[:-1]:
        (next_x, next_y, next_xdot, _) = slip.apex_map(
            slip_apex_x[-1], slip_apex_y[-1], slip_apex_xdot[-1], theta)
        slip_apex_x.append(next_x)
        slip_apex_y.append(next_y)
        slip_apex_xdot.append(next_xdot)
    sol = slip.simulate(np.concatenate((x0.numpy(), np.array([0.]))),
                        theta_step)
    slip_x = []
    slip_y = []
    for step in range(int(len(sol)/2)):
        # for step in range(int(len(theta_step)/2)):
        slip_x.extend(sol[2 * step].y[0])
        slip_y.extend(sol[2 * step].y[1])
        # slip_x.append(slip_apex_x[step])
        # slip_y.append(slip_apex_y[step])
        slip_r = np.array(sol[2 * step + 1].y[0])
        slip_theta = np.array(sol[2 * step + 1].y[1])
        slip_x_foot = np.array(sol[2 * step + 1].y[4])
        slip_x.extend(list(slip_x_foot - slip_r * np.sin(slip_theta)))
        slip_y.extend(list(slip_r * np.cos(slip_theta)))
    x_traj = torch.Tensor([slip_x, slip_y]).type(x0.dtype)
    x_traj_apex = torch.Tensor([slip_apex_x, slip_apex_y]).type(x0.dtype)
    return x_traj, x_traj_apex


def slip_nonlinear_traj(slip, x0, utraj):
    x_traj_nl = torch.Tensor(0, 3).type(x0.dtype)
    x_traj_nl = torch.cat((x_traj_nl, x0.unsqueeze(0)), axis=0)
    for i in range(utraj.shape[1] - 1):
        (next_x, next_y, next_xdot, _) = slip.apex_map(
            x_traj_nl[-1, 0], x_traj_nl[-1, 1], x_traj_nl[-1, 2], utraj[0, i])
        if next_x is None:
            return None
        x_traj_nl = torch.cat((x_traj_nl, torch.Tensor(
            [[next_x, next_y, next_xdot]]).type(x0.dtype)), axis=0)
    return x_traj_nl.t()
