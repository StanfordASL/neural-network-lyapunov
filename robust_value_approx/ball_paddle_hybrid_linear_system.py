import torch
import robust_value_approx.hybrid_linear_system as hybrid_linear_system


def get_ball_paddle_hybrid_linear_system(dtype, dt, x_lo, x_up, u_lo, u_up, 
                                         cr=.8, collision_eps=1e-4, 
                                         trapz=False):
    """
    x = [ballx, bally, paddley, ballvx, ballvy, paddlevy]
    u = [paddletheta, paddlevy_dot]
    @param cr The coefficient of restitution
    @param collision_eps The distance at which we consider collision
    @param trapz Whether or not to use trapezoidal integration
    """
    g = -9.81
    x_dim = 6
    u_dim = 2
    assert(len(x_lo) == x_dim)
    assert(len(x_up) == x_dim)
    assert(len(u_lo) == u_dim)
    assert(len(u_up) == u_dim)
    P_xu_lo = -torch.eye(x_dim + u_dim, dtype=dtype)
    P_xu_up = torch.eye(x_dim + u_dim, dtype=dtype)
    P_lim = torch.cat((P_xu_lo, P_xu_up), dim=0)
    q_lim = torch.cat((-x_lo, -u_lo, x_up, u_up), dim=0)
    hls = hybrid_linear_system.HybridLinearSystem(x_dim, u_dim, dtype)

    # free falling mode
    if trapz:
        X = torch.Tensor([[1., 0., 0., -.5*dt, 0., 0.],
                          [0., 1., 0., 0., -.5*dt, 0.],
                          [0., 0., 1., 0., 0., -.5*dt],
                          [0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0., 1.]]).type(dtype)
        Xinv = torch.inverse(X)
        A = Xinv @ torch.Tensor([[1., 0., 0., .5*dt, 0., 0.],
                                 [0., 1., 0., 0., .5*dt, 0.],
                                 [0., 0., 1., 0., 0., .5*dt],
                                 [0., 0., 0., 1., 0., 0.],
                                 [0., 0., 0., 0., 1., 0.],
                                 [0., 0., 0., 0., 0., 1.]]).type(dtype)
        B = Xinv @ torch.Tensor([[0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., dt]]).type(dtype)
        c = Xinv @ torch.Tensor([0., 0., 0., 0., dt*g, 0.]).type(dtype)        
    else:
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
        c = torch.Tensor([0., 0., 0., 0., dt*g, 0.]).type(dtype)
    # bally > paddley + collision_eps
    P = torch.Tensor([[0., -1., 1., 0., -dt, dt, 0., 0.]]).type(dtype)
    q = torch.Tensor([-collision_eps]).type(dtype)
    hls.add_mode(A, B, c, torch.cat(
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
    hls.add_mode(A, B, c, torch.cat(
        (P, P_lim), dim=0), torch.cat(
            (q, q_lim), dim=0))

    return hls
