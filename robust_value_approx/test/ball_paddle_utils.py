import robust_value_approx.value_to_optimization as value_to_optimization
import robust_value_approx.ball_paddle_hybrid_linear_system as bp

import torch
import numpy as np


def get_value_function(N=8):
    dtype = torch.float64
    dt = .1
    ball_capture = torch.Tensor([.5, .5]).type(dtype)
    capture_size = .05
    paddle_ang = torch.Tensor([.1, .2, .3, .4, .5]).type(dtype)
    x_lo = torch.Tensor([-10., -10., 0., -np.pi, -1e2, -1e2, 0.]).type(dtype)
    x_up = torch.Tensor([10., 10., 1., np.pi, 1e2, 1e2, 1e2]).type(dtype)
    u_lo = torch.Tensor([-1e3, -1e3]).type(dtype)
    u_up = torch.Tensor([1e3, 1e3]).type(dtype)
    b_cap_lo = ball_capture - capture_size
    b_cap_up = ball_capture + capture_size
    sys_con = bp.get_ball_paddle_hybrid_linear_system_vel_ctrl
    sys = sys_con(dtype, dt,
                  x_lo, x_up,
                  u_lo, u_up,
                  ball_capture_lo=b_cap_lo,
                  ball_capture_up=b_cap_up,
                  paddle_angles=paddle_ang,
                  cr=.9, collision_eps=.01,
                  midpoint=True)
    vf = value_to_optimization.ValueFunction(sys, N, x_lo, x_up, u_lo, u_up)
    Q = torch.diag(torch.Tensor([1., 1., 0., 0., 0., 0., 0.]).type(dtype))
    Qt = torch.diag(torch.Tensor([10., 10., 0., 0., 0., 0., 0.]).type(dtype))
    R = torch.diag(torch.Tensor([.1, .01]).type(dtype))
    Rt = torch.diag(torch.Tensor([.1, .01]).type(dtype))
    xtraj = torch.Tensor([ball_capture[0], ball_capture[1], 0., 0.,
                          0., 0., 0.]).type(dtype).unsqueeze(1).repeat(1, N-1)
    vf.set_cost(Q=Q, R=R)
    vf.set_terminal_cost(Qt=Qt, Rt=Rt)
    vf.set_traj(xtraj=xtraj)
    return vf


def get_value_function_vertical(N=10):
    dtype = torch.float64
    dt = .1
    ball_target = torch.Tensor([2.]).type(dtype)
    x_lo = torch.Tensor([0., 0., -100, -100]).type(dtype)
    x_up = torch.Tensor([5., 1., 100, 100]).type(dtype)
    u_lo = torch.Tensor([-1e4]).type(dtype)
    u_up = torch.Tensor([1e4]).type(dtype)
    sys_con = bp.get_ball_paddle_hybrid_linear_system_vertical
    sys = sys_con(dtype, dt,
                  x_lo, x_up,
                  u_lo, u_up,
                  cr=.9, collision_eps=.01,
                  midpoint=True)
    vf = value_to_optimization.ValueFunction(sys, N, x_lo, x_up, u_lo, u_up)
    Q = torch.diag(torch.Tensor([1., 0., 0., 0.]).type(dtype))
    Qt = torch.diag(torch.Tensor([100., 0., 0., 0.]).type(dtype))
    R = torch.diag(torch.Tensor([.001]).type(dtype))
    Rt = torch.diag(torch.Tensor([.001]).type(dtype))
    xtraj = torch.Tensor(
        [ball_target[0], 0., 0., 0.]).type(dtype).unsqueeze(1).repeat(1, N-1)
    vf.set_cost(Q=Q, R=R)
    vf.set_terminal_cost(Qt=Qt, Rt=Rt)
    vf.set_traj(xtraj=xtraj)
    return vf
