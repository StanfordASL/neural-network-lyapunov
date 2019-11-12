import torch
import hybrid_linear_system


def get_ball_paddle_hybrid_linear_system(dtype, dt, x_lo, x_up, u_lo, u_up):
    """
    x = [ballx, bally, paddley, ballvx, ballvy, paddlevy]
    u = [paddletheta, paddlevy_dot]
    """
    g = -9.81
    x_dim = 6
    u_dim = 2

    assert(len(x_lo) == x_dim)
    assert(len(x_up) == x_dim)
    assert(len(u_lo) == u_dim)
    assert(len(u_up) == u_dim)

    # distance from surface considered in collision
    collision_eps = 1e-4
    # coefficient of restitution
    cr = .8

    P_xu_lo = -torch.eye(x_dim + u_dim, dtype=dtype)
    P_xu_up = torch.eye(x_dim + u_dim, dtype=dtype)
    P_lim = torch.cat((P_xu_lo, P_xu_up), dim=0)
    q_lim = torch.cat((-x_lo, -u_lo, x_up, u_up), dim=0)

    hls = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)

    # free falling mode
    A = torch.Tensor([[1., 0., 0., dt, 0., 0.],
                      [0., 1., 0., 0., dt, 0.],
                      [0., 0., 1., 0., 0., dt],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.]]).type(dtype)
    B = torch.Tensor([[0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [0., dt]]).type(dtype)
    c = torch.Tensor([0., 0., 0., 0., g, 0.]).type(dtype)
    # bally > paddley + collision_eps
    P = torch.Tensor([[0., -1., 1., 0., -dt, dt, 0., 0.]]).type(dtype)
    q = torch.Tensor([-collision_eps]).type(dtype)
    hls.add_mode(
        A, B, c, torch.cat(
            (P, P_lim), dim=0), torch.cat(
            (q, q_lim), dim=0))

    # colliding with the paddle
    A = torch.Tensor([[1., 0., 0., dt, 0., 0.],
                      [0., 0., 1., 0., 0., dt],
                      [0., 0., 1., 0., 0., dt],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., -cr, 1. + cr],
                      [0., 0., 0., 0., 0., 1.]]).type(dtype)
    B = torch.Tensor([[0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [0., dt],
                      [0., dt]]).type(dtype)
    c = torch.Tensor([0., collision_eps, 0., 0., 0., 0.]).type(dtype)
    # bally <= paddley + collision_eps
    P = torch.Tensor([[0., 1., -1., 0., dt, -dt, 0., 0.]]).type(dtype)
    q = torch.Tensor([collision_eps]).type(dtype)
    hls.add_mode(
        A, B, c, torch.cat(
            (P, P_lim), dim=0), torch.cat(
            (q, q_lim), dim=0))

    return hls
